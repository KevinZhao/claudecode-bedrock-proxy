#!/usr/bin/env python3
"""Bedrock Effort Max Proxy - forces adaptive thinking + effort=max for Opus/Sonnet 4.6,
maximizes budget_tokens for older models, and injects prompt caching breakpoints."""

import base64
import itertools
import json
import logging
import os
import time
from typing import Optional
from urllib.parse import quote, unquote

import aiohttp
from aiohttp import web
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.session import Session as BotocoreSession
from yarl import URL

# Configuration
LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = int(os.environ.get("PROXY_PORT", "8888"))
TARGET_REGION = os.environ.get("AWS_REGION", "ap-northeast-1")
RUNTIME_HOST = f"bedrock-runtime.{TARGET_REGION}.amazonaws.com"
CONTROL_HOST = f"bedrock.{TARGET_REGION}.amazonaws.com"

# Control plane path prefixes (route to bedrock instead of bedrock-runtime)
CONTROL_PLANE_PREFIXES = ("/inference-profiles", "/foundation-models", "/guardrails")

# Anthropic API model ID -> Bedrock cross-region model ID
MODEL_MAP = {
    "claude-opus-4-6": "global.anthropic.claude-opus-4-6-v1",
    "claude-sonnet-4-6": "global.anthropic.claude-sonnet-4-6",
    "claude-sonnet-4-5": "global.anthropic.claude-sonnet-4-6",
    "claude-sonnet-4-5-20250929": "global.anthropic.claude-sonnet-4-6",
    "claude-opus-4-5": "global.anthropic.claude-opus-4-6-v1",
    "claude-opus-4-5-20251101": "global.anthropic.claude-opus-4-6-v1",
    "claude-haiku-4-5": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-haiku-4-5-20251001": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-sonnet-4-20250514": "global.anthropic.claude-sonnet-4-20250514-v1:0",
}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bedrock-proxy")

# Auth: bearer token (preferred) or SigV4 fallback
_bearer_token = os.environ.get("BEDROCK_BEARER_TOKEN")
_credentials = None
if not _bearer_token:
    _botocore_session = BotocoreSession()
    _credentials = _botocore_session.get_credentials()
AUTH_MODE = "bearer" if _bearer_token else "sigv4"

# Model classification for thinking strategy
# Opus 4.6 / Sonnet 4.6: adaptive thinking + effort=max (budget_tokens deprecated)
ADAPTIVE_MODELS = ("opus-4-6", "sonnet-4-6")
# Older models: manual thinking with budget_tokens
# Haiku: no thinking support
THINKING_SKIP_MODELS = ("haiku",)
# Default max budget for older models when max_tokens is absent
DEFAULT_MAX_BUDGET = 128000

# Prompt caching configuration
CACHE_ENABLED = os.environ.get("CACHE_ENABLED", "1") == "1"
CACHE_TTL = os.environ.get("CACHE_TTL", "1h")  # "5m" or "1h"

# Stats
_req_counter = itertools.count(1)
_request_count = 0
_thinking_modified = 0
_cache_injected = 0
_cache_read_total = 0
_cache_write_total = 0

# Shared upstream session (created on first request)
_upstream_session: Optional[aiohttp.ClientSession] = None
_UPSTREAM_TIMEOUT = aiohttp.ClientTimeout(total=600, sock_read=600)


async def _get_session() -> aiohttp.ClientSession:
    global _upstream_session
    if _upstream_session is None or _upstream_session.closed:
        connector = aiohttp.TCPConnector(limit=20, keepalive_timeout=60)
        _upstream_session = aiohttp.ClientSession(
            timeout=_UPSTREAM_TIMEOUT, connector=connector
        )
    return _upstream_session


MAX_CACHE_BREAKPOINTS = 4


def _iter_cache_control_blocks(data):
    """Yield all dicts that have a 'cache_control' key in tools, system, and messages."""
    for tool in (data.get("tools") or []):
        if isinstance(tool, dict) and "cache_control" in tool:
            yield tool
    system = data.get("system")
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and "cache_control" in block:
                yield block
    for msg in (data.get("messages") or []):
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "cache_control" in block:
                    yield block


def _count_existing_cache_control(data):
    return sum(1 for _ in _iter_cache_control_blocks(data))


def _upgrade_existing_ttl(data, ttl):
    count = 0
    for block in _iter_cache_control_blocks(data):
        block["cache_control"]["ttl"] = ttl
        count += 1
    return count


def _inject_cache_control(data):
    """Inject cache_control breakpoints on tools, system, and last assistant message.
    Respects the API limit of 4 total cache_control blocks (including pre-existing ones).
    Returns (count_added, description_string)."""
    marker = {"type": "ephemeral"}
    if CACHE_TTL != "5m":
        marker["ttl"] = CACHE_TTL

    existing = _count_existing_cache_control(data)

    # Upgrade TTL on all pre-existing cache_control markers
    upgraded = 0
    if CACHE_TTL != "5m":
        upgraded = _upgrade_existing_ttl(data, CACHE_TTL)

    budget = MAX_CACHE_BREAKPOINTS - existing
    added = 0
    parts = []

    if budget <= 0:
        if upgraded > 0:
            return 0, f"ttl-upgrade({upgraded}->{CACHE_TTL},existing={existing})"
        return 0, f"no-op(existing={existing})"

    # 1. Tools -- inject on last tool definition
    tools = data.get("tools")
    if tools and isinstance(tools, list) and len(tools) > 0 and added < budget:
        if "cache_control" not in tools[-1]:
            tools[-1]["cache_control"] = dict(marker)
            added += 1
            parts.append("tools")

    # 2. System prompt -- convert string->array if needed, inject on last block
    system = data.get("system")
    if system and added < budget:
        if isinstance(system, str):
            data["system"] = [{"type": "text", "text": system, "cache_control": dict(marker)}]
            added += 1
            parts.append("system")
        elif isinstance(system, list) and len(system) > 0:
            last = system[-1]
            if isinstance(last, dict) and "cache_control" not in last:
                last["cache_control"] = dict(marker)
                added += 1
                parts.append("system")

    # 3. Messages -- last assistant turn's last non-thinking block
    if added < budget:
        messages = data.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, str):
                    msg["content"] = [{"type": "text", "text": content,
                                       "cache_control": dict(marker)}]
                    added += 1
                    parts.append("msgs")
                elif isinstance(content, list):
                    for block in reversed(content):
                        if isinstance(block, dict) and block.get("type") not in ("thinking", "redacted_thinking"):
                            if "cache_control" not in block:
                                block["cache_control"] = dict(marker)
                                added += 1
                                parts.append("msgs")
                            break
                break

    upg = f",upg={upgraded}" if upgraded > 0 else ""
    desc = f"{added}bp({'+'.join(parts)},{CACHE_TTL},pre={existing}{upg})" if added > 0 else f"no-op(existing={existing}{upg})"
    return added, desc


# Pre-computed base64 prefixes for quick EventStream filtering
# b'{"type":"message_start"' -> base64 starts with "eyJ0eXBlIjoibWVzc2FnZV9zdGFydC"
# b'{"type":"message_delta"' -> base64 starts with "eyJ0eXBlIjoibWVzc2FnZV9kZWx0YS"
_B64_MESSAGE_START = b"eyJ0eXBlIjoibWVzc2FnZV9zdGFydC"
_B64_MESSAGE_DELTA = b"eyJ0eXBlIjoibWVzc2FnZV9kZWx0YS"


def _extract_cache_metrics(chunk_bytes, req_id=0):
    """Extract cache metrics from Bedrock EventStream binary frames.

    Bedrock streaming format:
    - Frame: [total_len:4][headers_len:4][prelude_crc:4][headers][payload][msg_crc:4]
    - Payload JSON: {"bytes": "<base64-encoded-anthropic-event>"}
    - Base64-decoded = actual Anthropic SSE JSON (message_start, message_delta, etc.)
    """
    # Quick check: skip chunks that can't contain message_start or message_delta
    if _B64_MESSAGE_START not in chunk_bytes and _B64_MESSAGE_DELTA not in chunk_bytes:
        return None
    try:
        buf = chunk_bytes
        while len(buf) >= 16:
            total_len = int.from_bytes(buf[0:4], "big")
            if total_len < 16 or total_len > len(buf):
                break
            headers_len = int.from_bytes(buf[4:8], "big")
            payload_start = 12 + headers_len
            payload_end = total_len - 4
            if payload_start < payload_end:
                payload = buf[payload_start:payload_end]
                text = payload.decode("utf-8", errors="ignore").strip()
                if text.startswith("{"):
                    try:
                        wrapper = json.loads(text)
                        b64 = wrapper.get("bytes")
                        if b64:
                            inner = base64.b64decode(b64).decode("utf-8", errors="ignore")
                            obj = json.loads(inner)
                        else:
                            obj = wrapper
                        evt_type = obj.get("type", "")
                        usage = None
                        if evt_type == "message_start":
                            usage = (obj.get("message") or {}).get("usage")
                        elif evt_type == "message_delta":
                            usage = obj.get("usage")
                        if usage:
                            cr = usage.get("cache_read_input_tokens", 0)
                            cw = usage.get("cache_creation_input_tokens", 0)
                            inp = usage.get("input_tokens", 0)
                            cc = usage.get("cache_creation", {})
                            ttl_info = ""
                            if cc:
                                t5 = cc.get("ephemeral_5m_input_tokens", 0)
                                t1h = cc.get("ephemeral_1h_input_tokens", 0)
                                if t5 > 0 or t1h > 0:
                                    ttl_info = f" [5m={t5},1h={t1h}]"
                            log.info(f"[#{req_id}] cache-extract: {evt_type} cr={cr} cw={cw} inp={inp}{ttl_info}")
                            if cr > 0 or cw > 0:
                                return cr, cw, inp
                    except (json.JSONDecodeError, ValueError):
                        pass
            buf = buf[total_len:]
    except Exception as e:
        log.debug(f"[#{req_id}] cache-extract error: {e}")
    return None


async def health(request):
    return web.json_response({
        "status": "ok",
        "region": TARGET_REGION,
        "runtime": RUNTIME_HOST,
        "control": CONTROL_HOST,
        "requests_served": _request_count,
        "thinking_modified": _thinking_modified,
        "cache_enabled": CACHE_ENABLED,
        "cache_ttl": CACHE_TTL,
        "cache_injected": _cache_injected,
        "cache_read_tokens": _cache_read_total,
        "cache_write_tokens": _cache_write_total,
    })


def _resolve_target(raw_path):
    """Route to control plane or runtime host, and remap model IDs if needed."""
    decoded_path = unquote(raw_path)

    # Control plane API -> bedrock.{region}
    if any(decoded_path.startswith(p) for p in CONTROL_PLANE_PREFIXES):
        return CONTROL_HOST, raw_path, None

    # Runtime API: check if model ID needs Bedrock format mapping
    if "/model/" in decoded_path:
        parts = decoded_path.split("/model/", 1)
        rest = parts[1]  # e.g. "claude-sonnet-4-5/invoke-with-response-stream"
        model_id = rest.split("/", 1)[0]
        suffix = rest.split("/", 1)[1] if "/" in rest else ""

        bedrock_id = MODEL_MAP.get(model_id)
        if bedrock_id:
            new_path = f"/model/{quote(bedrock_id, safe='/:@')}/{suffix}"
            log.info(f"  model remap: {model_id} -> {bedrock_id}")
            return RUNTIME_HOST, new_path, bedrock_id

        return RUNTIME_HOST, raw_path, model_id

    return RUNTIME_HOST, raw_path, None


def _modify_thinking(data, model_lower):
    """Modify thinking/effort settings. Returns (modified: bool, action_desc: str)."""
    skip = any(s in model_lower for s in THINKING_SKIP_MODELS)
    is_adaptive = any(s in model_lower for s in ADAPTIVE_MODELS)

    if skip:
        return False, "skip (haiku)"

    if "messages" not in data:
        return False, "passthrough"

    if is_adaptive:
        return _modify_adaptive_thinking(data)
    return _modify_legacy_thinking(data)


def _modify_adaptive_thinking(data):
    """Opus 4.6 / Sonnet 4.6: adaptive thinking + effort=max + 1M context beta."""
    thinking = data.get("thinking")
    effort = (data.get("output_config") or {}).get("effort")
    changes = []

    if not isinstance(thinking, dict) or thinking.get("type") != "adaptive":
        data["thinking"] = {"type": "adaptive"}
        changes.append(f"thinking->adaptive (was {thinking})")

    if effort != "max":
        if "output_config" not in data:
            data["output_config"] = {}
        data["output_config"]["effort"] = "max"
        changes.append(f"effort->max (was {effort})")

    # Inject 1M context beta header
    context_beta = "context-1m-2025-08-07"
    betas = data.get("anthropic_beta", [])
    if context_beta not in betas:
        betas.append(context_beta)
        data["anthropic_beta"] = betas
        changes.append(f"beta+={context_beta}")

    if changes:
        return True, "; ".join(changes)
    return False, "already adaptive+max"


def _modify_legacy_thinking(data):
    """Older models (Sonnet 4.5, Opus 4.5, etc.): budget_tokens."""
    max_tokens = data.get("max_tokens", DEFAULT_MAX_BUDGET)
    target_budget = max_tokens - 1
    thinking = data.get("thinking")

    if thinking is None or (isinstance(thinking, dict) and thinking.get("type") == "disabled"):
        data["thinking"] = {
            "type": "enabled",
            "budget_tokens": target_budget,
        }
        betas = data.get("anthropic_beta", [])
        thinking_beta = "interleaved-thinking-2025-05-14"
        if thinking_beta not in betas:
            betas.append(thinking_beta)
            data["anthropic_beta"] = betas
        return True, f"injected budget={target_budget}"

    if isinstance(thinking, dict):
        old_budget = thinking.get("budget_tokens", 0)
        if old_budget < target_budget:
            thinking["budget_tokens"] = target_budget
            return True, f"upgraded {old_budget}->{target_budget}"
        return False, f"already max budget={old_budget}"

    return False, "skip (unexpected format)"


async def proxy_handler(request: web.Request):
    global _request_count, _thinking_modified, _cache_injected, _cache_read_total, _cache_write_total
    req_id = next(_req_counter)
    _request_count += 1

    raw_path = request.url.raw_path
    query = request.query_string
    method = request.method

    target_host, resolved_path, model_id = _resolve_target(raw_path)

    target_url_str = f"https://{target_host}{resolved_path}"
    if query:
        target_url_str += f"?{query}"

    body = await request.read()

    decoded_path = unquote(resolved_path)
    if not model_id and "/model/" in decoded_path:
        model_id = decoded_path.split("/model/", 1)[1].split("/")[0]

    is_invoke = "/invoke" in decoded_path
    model_lower = (model_id or "").lower()
    thinking_action = "passthrough"
    cache_action = "off"

    if body and is_invoke:
        try:
            data = json.loads(body)
            modified = False

            # Strip defer_loading from tools (client-side field, Bedrock rejects it)
            tools = data.get("tools")
            if tools and isinstance(tools, list):
                for t in tools:
                    if isinstance(t, dict) and "defer_loading" in t:
                        del t["defer_loading"]
                        modified = True

            # Thinking/effort modification
            thinking_modified, thinking_action = _modify_thinking(data, model_lower)
            if thinking_modified:
                modified = True
                _thinking_modified += 1

            # Prompt caching injection + TTL upgrade
            if CACHE_ENABLED and "messages" in data:
                cache_added, cache_action = _inject_cache_control(data)
                if cache_added > 0 or "ttl-upgrade" in cache_action:
                    modified = True
                if cache_added > 0:
                    _cache_injected += 1

            if modified:
                body = json.dumps(data, separators=(",", ":")).encode()

            log.info(
                f"[#{req_id}] -> {method} model={model_id or 'n/a'} | "
                f"thinking: {thinking_action} | cache: {cache_action} | "
                f"body={len(body)}B -> {target_host}"
            )
        except (json.JSONDecodeError, KeyError) as e:
            log.warning(f"[#{req_id}] Cannot parse body: {e}")
            log.info(f"[#{req_id}] -> {method} {decoded_path} (passthrough) -> {target_host}")
    else:
        log.info(f"[#{req_id}] -> {method} {decoded_path} -> {target_host}")

    # Build headers for upstream request
    if _bearer_token:
        signed_headers = {
            "Content-Type": request.content_type or "application/json",
            "Authorization": f"Bearer {_bearer_token}",
        }
    else:
        headers = {
            "Content-Type": request.content_type or "application/json",
            "Host": target_host,
        }
        aws_request = AWSRequest(method=method, url=target_url_str, data=body, headers=headers)
        SigV4Auth(_credentials, "bedrock", TARGET_REGION).add_auth(aws_request)
        signed_headers = dict(aws_request.headers)

    target_url_yarl = URL(target_url_str, encoded=True)
    is_streaming = "response-stream" in resolved_path
    t0 = time.monotonic()

    try:
        session = await _get_session()
        async with session.request(
            method, target_url_yarl, headers=signed_headers, data=body, ssl=True,
        ) as resp:
            elapsed = time.monotonic() - t0

            if resp.status >= 400:
                err_body = await resp.read()
                log.error(f"[#{req_id}] <- {resp.status} in {elapsed:.1f}s | {err_body[:500]}")
                return web.Response(
                    status=resp.status, body=err_body, content_type=resp.content_type,
                )

            if is_streaming:
                return await _stream_response(request, resp, req_id, t0)

            resp_body = await resp.read()
            log.info(f"[#{req_id}] <- {resp.status} {len(resp_body)}B in {elapsed:.1f}s")
            return web.Response(
                status=resp.status, body=resp_body, content_type=resp.content_type,
            )
    except Exception as e:
        elapsed = time.monotonic() - t0
        log.error(f"[#{req_id}] Proxy error after {elapsed:.1f}s: {e}")
        return web.json_response({"error": "upstream_error"}, status=502)


async def _stream_response(request, resp, req_id, t0):
    """Forward a streaming response, extracting cache metrics along the way."""
    global _cache_read_total, _cache_write_total
    response = web.StreamResponse(
        status=resp.status,
        headers={
            k: v
            for k, v in resp.headers.items()
            if k.lower() not in ("transfer-encoding", "content-encoding", "connection")
        },
    )
    await response.prepare(request)

    total_bytes = 0
    cache_logged = False
    async for chunk in resp.content.iter_any():
        await response.write(chunk)
        total_bytes += len(chunk)
        if not cache_logged:
            metrics = _extract_cache_metrics(chunk, req_id)
            if metrics:
                cr, cw, inp = metrics
                cache_logged = True
                _cache_read_total += cr
                _cache_write_total += cw
                log.info(f"[#{req_id}] cache: read={cr} write={cw} uncached={inp}")

    await response.write_eof()
    elapsed = time.monotonic() - t0
    log.info(f"[#{req_id}] <- {resp.status} streamed {total_bytes}B in {elapsed:.1f}s")
    return response


async def _on_shutdown(app):
    global _upstream_session
    if _upstream_session and not _upstream_session.closed:
        await _upstream_session.close()
        _upstream_session = None


def create_app():
    app = web.Application(client_max_size=50 * 1024 * 1024)  # 50MB max body
    app.router.add_get("/health", health)
    app.router.add_route("*", "/{path_info:.*}", proxy_handler)
    app.on_shutdown.append(_on_shutdown)
    return app


if __name__ == "__main__":
    log.info(f"Bedrock Effort Max Proxy starting on {LISTEN_HOST}:{LISTEN_PORT}")
    log.info(f"Runtime: {RUNTIME_HOST}")
    log.info(f"Control: {CONTROL_HOST}")
    log.info(f"Auth: {AUTH_MODE}")
    log.info(f"Model remaps: {len(MODEL_MAP)} entries")
    log.info(f"Adaptive models (effort=max): {ADAPTIVE_MODELS}")
    log.info(f"Legacy models: budget_tokens maximized")
    log.info(f"Skip: {THINKING_SKIP_MODELS}")
    log.info(f"Cache: enabled={CACHE_ENABLED}, ttl={CACHE_TTL}")
    app = create_app()
    web.run_app(app, host=LISTEN_HOST, port=LISTEN_PORT, print=None)

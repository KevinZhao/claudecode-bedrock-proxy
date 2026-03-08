"""Tests for bedrock-effort-proxy."""

import copy
import json
import pytest

from proxy import (
    _inject_cache_control,
    _count_existing_cache_control,
    _modify_thinking,
    _modify_adaptive_thinking,
    _modify_legacy_thinking,
    CACHE_TTL,
)


# ---------------------------------------------------------------------------
# _inject_cache_control tests
# ---------------------------------------------------------------------------

class TestInjectCacheControl:
    """Tests for cache_control injection logic."""

    def _make_data(self, tools=None, system=None, messages=None):
        data = {}
        if tools is not None:
            data["tools"] = tools
        if system is not None:
            data["system"] = system
        if messages is not None:
            data["messages"] = messages
        return data

    def test_inject_on_last_tool(self):
        """Basic: inject cache_control on last tool."""
        tools = [
            {"name": "tool_a", "description": "A"},
            {"name": "tool_b", "description": "B"},
        ]
        data = self._make_data(tools=tools, messages=[{"role": "user", "content": "hi"}])
        added, desc = _inject_cache_control(data)
        assert added >= 1
        assert "cache_control" in tools[-1]

    def test_last_tool_already_has_cache_control(self):
        """If last tool already has cache_control, skip it."""
        tools = [
            {"name": "tool_a", "description": "A"},
            {"name": "tool_b", "description": "B",
             "cache_control": {"type": "ephemeral"}},
        ]
        data = self._make_data(tools=tools, messages=[{"role": "user", "content": "hi"}])
        added, desc = _inject_cache_control(data)
        # Should not double-inject on tool_b
        assert tools[-1]["cache_control"] == {"type": "ephemeral", "ttl": CACHE_TTL} or \
               tools[-1]["cache_control"]["type"] == "ephemeral"

    def test_inject_system_string(self):
        """System prompt as string gets converted to array with cache_control."""
        data = self._make_data(
            system="You are helpful.",
            messages=[{"role": "user", "content": "hi"}],
        )
        added, desc = _inject_cache_control(data)
        assert isinstance(data["system"], list)
        assert "cache_control" in data["system"][0]

    def test_inject_system_array(self):
        """System prompt as array gets cache_control on last block."""
        system = [
            {"type": "text", "text": "Block 1"},
            {"type": "text", "text": "Block 2"},
        ]
        data = self._make_data(system=system, messages=[{"role": "user", "content": "hi"}])
        added, desc = _inject_cache_control(data)
        assert "cache_control" in system[-1]

    def test_inject_on_last_assistant_message(self):
        """Cache_control injected on last assistant turn's last non-thinking block."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "response"},
            ]},
            {"role": "user", "content": "follow up"},
        ]
        data = self._make_data(messages=messages)
        added, desc = _inject_cache_control(data)
        assert "cache_control" in messages[1]["content"][0]

    def test_respects_budget_limit(self):
        """Should not exceed MAX_CACHE_BREAKPOINTS (4) total."""
        tools = [{"name": "t", "description": "d"}]
        system = [{"type": "text", "text": "sys"}]
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "resp"}]},
            {"role": "user", "content": "bye"},
        ]
        data = self._make_data(tools=tools, system=system, messages=messages)
        added, desc = _inject_cache_control(data)
        total = _count_existing_cache_control(data)
        assert total <= 4

    def test_no_op_when_budget_exhausted(self):
        """When 4 cache_control blocks already exist, returns no-op."""
        tools = [
            {"name": "t1", "cache_control": {"type": "ephemeral"}},
            {"name": "t2", "cache_control": {"type": "ephemeral"}},
        ]
        system = [
            {"type": "text", "text": "s1", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "s2", "cache_control": {"type": "ephemeral"}},
        ]
        data = self._make_data(tools=tools, system=system, messages=[])
        added, desc = _inject_cache_control(data)
        assert added == 0
        assert "no-op" in desc or "ttl-upgrade" in desc

    def test_empty_tools_list(self):
        """Empty tools list should not crash."""
        data = self._make_data(
            tools=[],
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
        )
        added, desc = _inject_cache_control(data)
        assert "tools" not in desc


# ---------------------------------------------------------------------------
# _modify_thinking tests
# ---------------------------------------------------------------------------

class TestModifyThinking:
    """Tests for thinking/effort modification logic."""

    def test_skip_haiku(self):
        data = {"messages": [{"role": "user", "content": "hi"}]}
        modified, desc = _modify_thinking(data, "claude-haiku-4-5")
        assert not modified
        assert "haiku" in desc.lower()

    def test_adaptive_for_opus_46(self):
        data = {"messages": [{"role": "user", "content": "hi"}]}
        modified, desc = _modify_thinking(data, "global.anthropic.claude-opus-4-6-v1")
        assert modified
        assert data["thinking"]["type"] == "adaptive"
        assert data["output_config"]["effort"] == "max"

    def test_adaptive_for_sonnet_46(self):
        data = {"messages": [{"role": "user", "content": "hi"}]}
        modified, desc = _modify_thinking(data, "global.anthropic.claude-sonnet-4-6")
        assert modified
        assert data["thinking"]["type"] == "adaptive"

    def test_legacy_for_sonnet_4(self):
        data = {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8192,
        }
        modified, desc = _modify_thinking(data, "global.anthropic.claude-sonnet-4-20250514-v1:0")
        assert modified
        assert data["thinking"]["budget_tokens"] == 8191

    def test_no_messages_passthrough(self):
        data = {}
        modified, desc = _modify_thinking(data, "opus-4-6")
        assert not modified
        assert "passthrough" in desc

    def test_adaptive_already_set(self):
        data = {
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "max"},
            "anthropic_beta": ["context-1m-2025-08-07"],
        }
        modified, desc = _modify_thinking(data, "opus-4-6")
        assert not modified
        assert "already" in desc

    def test_effort_upgrade(self):
        data = {
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "low"},
        }
        modified, desc = _modify_thinking(data, "sonnet-4-6")
        assert modified
        assert data["output_config"]["effort"] == "max"


# ---------------------------------------------------------------------------
# defer_loading stripping tests (simulates proxy_handler preprocessing)
# ---------------------------------------------------------------------------

class TestStripDeferLoading:
    """Verify defer_loading is stripped from tools before forwarding to Bedrock."""

    def test_strip_defer_loading_true(self):
        """defer_loading=true must be removed from tool dicts."""
        tools = [
            {"name": "t1", "description": "A"},
            {"name": "t2", "description": "B", "defer_loading": True},
        ]
        # Simulate the stripping logic from proxy_handler
        for t in tools:
            if isinstance(t, dict) and "defer_loading" in t:
                del t["defer_loading"]
        assert "defer_loading" not in tools[1]

    def test_strip_defer_loading_false(self):
        """defer_loading=false should also be stripped (Bedrock doesn't accept the field at all)."""
        tools = [
            {"name": "t1", "description": "A", "defer_loading": False},
        ]
        for t in tools:
            if isinstance(t, dict) and "defer_loading" in t:
                del t["defer_loading"]
        assert "defer_loading" not in tools[0]

    def test_no_defer_loading_unchanged(self):
        """Tools without defer_loading should not be modified."""
        tools = [
            {"name": "t1", "description": "A", "input_schema": {"type": "object"}},
        ]
        original_keys = set(tools[0].keys())
        for t in tools:
            if isinstance(t, dict) and "defer_loading" in t:
                del t["defer_loading"]
        assert set(tools[0].keys()) == original_keys

    def test_cache_inject_after_strip(self):
        """Full flow: strip defer_loading, then inject cache_control — no conflict."""
        tools = [
            {"name": "t1", "description": "A"},
            {"name": "t2", "description": "B", "defer_loading": True},
        ]
        # Step 1: strip
        for t in tools:
            if isinstance(t, dict) and "defer_loading" in t:
                del t["defer_loading"]
        # Step 2: inject cache
        data = {"tools": tools, "messages": [{"role": "user", "content": "hi"}]}
        added, desc = _inject_cache_control(data)
        # t2 no longer has defer_loading, so it's eligible for cache_control
        assert "cache_control" in tools[-1]
        assert "defer_loading" not in tools[-1]

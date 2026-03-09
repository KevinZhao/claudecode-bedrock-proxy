package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	v4 "github.com/aws/aws-sdk-go-v2/aws/signer/v4"
	"github.com/aws/aws-sdk-go-v2/config"
)

// --- Configuration ---

var (
	listenAddr   = envOr("LISTEN_HOST", "127.0.0.1") + ":" + envOr("PROXY_PORT", "8888")
	targetRegion = envOr("AWS_REGION", "ap-northeast-1")
	runtimeHost  = "bedrock-runtime." + targetRegion + ".amazonaws.com"
	controlHost  = "bedrock." + targetRegion + ".amazonaws.com"
	bearerToken  = os.Getenv("BEDROCK_BEARER_TOKEN")
	cacheEnabled = envOr("CACHE_ENABLED", "1") == "1"
	cacheTTL     = envOr("CACHE_TTL", "1h")
)

var controlPrefixes = []string{"/inference-profiles", "/foundation-models", "/guardrails"}

var modelMap = map[string]string{
	"claude-opus-4-6":            "global.anthropic.claude-opus-4-6-v1",
	"claude-sonnet-4-6":          "global.anthropic.claude-sonnet-4-6",
	"claude-sonnet-4-5":          "global.anthropic.claude-sonnet-4-6",
	"claude-sonnet-4-5-20250929": "global.anthropic.claude-sonnet-4-6",
	"claude-opus-4-5":            "global.anthropic.claude-opus-4-6-v1",
	"claude-opus-4-5-20251101":   "global.anthropic.claude-opus-4-6-v1",
	"claude-haiku-4-5":           "global.anthropic.claude-haiku-4-5-20251001-v1:0",
	"claude-haiku-4-5-20251001":  "global.anthropic.claude-haiku-4-5-20251001-v1:0",
	"claude-sonnet-4-20250514":   "global.anthropic.claude-sonnet-4-20250514-v1:0",
}

var (
	adaptiveModels     = []string{"opus-4-6", "sonnet-4-6"}
	thinkingSkipModels = []string{"haiku"}
)

const (
	defaultMaxBudget    = 128000
	maxCacheBreakpoints = 4
)

// --- Stats ---

var (
	reqCounter      atomic.Int64
	requestCount    atomic.Int64
	thinkingModCnt  atomic.Int64
	cacheInjCnt     atomic.Int64
	cacheReadTotal  atomic.Int64
	cacheWriteTotal atomic.Int64
)

// --- AWS ---

var (
	awsCredProvider aws.CredentialsProvider
	awsSigner       *v4.Signer
)

// --- HTTP Client (h2-enabled) ---

var httpClient = &http.Client{
	Transport: &http.Transport{
		ForceAttemptHTTP2:     true,
		MaxIdleConnsPerHost:   20,
		IdleConnTimeout:       60 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ResponseHeaderTimeout: 600 * time.Second,
	},
}

// --- Helpers ---

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func strOr(s, def string) string {
	if s == "" {
		return def
	}
	return s
}

func containsAny(s string, subs []string) bool {
	for _, sub := range subs {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}

func sliceHas(slice []any, val string) bool {
	for _, v := range slice {
		if s, ok := v.(string); ok && s == val {
			return true
		}
	}
	return false
}

// Type-safe accessors for map[string]any (JSON objects)

func jMap(data map[string]any, key string) map[string]any {
	if v, ok := data[key].(map[string]any); ok {
		return v
	}
	return nil
}

func jSlice(data map[string]any, key string) []any {
	if v, ok := data[key].([]any); ok {
		return v
	}
	return nil
}

func jStr(data map[string]any, key string) string {
	if v, ok := data[key].(string); ok {
		return v
	}
	return ""
}

func jNum(data map[string]any, key string) float64 {
	if v, ok := data[key].(float64); ok {
		return v
	}
	return 0
}

// --- Route Resolution ---

func resolveTarget(rawPath string) (host, path, modelID string) {
	decoded, _ := url.PathUnescape(rawPath)

	for _, prefix := range controlPrefixes {
		if strings.HasPrefix(decoded, prefix) {
			return controlHost, rawPath, ""
		}
	}

	if idx := strings.Index(decoded, "/model/"); idx >= 0 {
		rest := decoded[idx+len("/model/"):]
		slash := strings.IndexByte(rest, '/')
		var mid, suffix string
		if slash >= 0 {
			mid = rest[:slash]
			suffix = rest[slash+1:]
		} else {
			mid = rest
		}

		if bedrockID, ok := modelMap[mid]; ok {
			escaped := url.PathEscape(bedrockID)
			escaped = strings.ReplaceAll(escaped, "%3A", ":")
			newPath := "/model/" + escaped + "/" + suffix
			log.Printf("  model remap: %s -> %s", mid, bedrockID)
			return runtimeHost, newPath, bedrockID
		}
		return runtimeHost, rawPath, mid
	}

	return runtimeHost, rawPath, ""
}

// --- Thinking Modification ---

func modifyThinking(data map[string]any, modelLower string) (bool, string) {
	if containsAny(modelLower, thinkingSkipModels) {
		return false, "skip (haiku)"
	}
	if _, ok := data["messages"]; !ok {
		return false, "passthrough"
	}
	if containsAny(modelLower, adaptiveModels) {
		return modifyAdaptiveThinking(data)
	}
	return modifyLegacyThinking(data)
}

func modifyAdaptiveThinking(data map[string]any) (bool, string) {
	var changes []string

	thinking := jMap(data, "thinking")
	if thinking == nil || jStr(thinking, "type") != "adaptive" {
		old := fmt.Sprintf("%v", data["thinking"])
		data["thinking"] = map[string]any{"type": "adaptive"}
		changes = append(changes, "thinking->adaptive (was "+old+")")
	}

	oc := jMap(data, "output_config")
	effort := ""
	if oc != nil {
		effort = jStr(oc, "effort")
	}
	if effort != "max" {
		if oc == nil {
			oc = map[string]any{}
			data["output_config"] = oc
		}
		oc["effort"] = "max"
		changes = append(changes, fmt.Sprintf("effort->max (was %s)", effort))
	}

	const contextBeta = "context-1m-2025-08-07"
	betas := jSlice(data, "anthropic_beta")
	if !sliceHas(betas, contextBeta) {
		betas = append(betas, contextBeta)
		data["anthropic_beta"] = betas
		changes = append(changes, "beta+="+contextBeta)
	}

	if len(changes) > 0 {
		return true, strings.Join(changes, "; ")
	}
	return false, "already adaptive+max"
}

func modifyLegacyThinking(data map[string]any) (bool, string) {
	maxTokens := defaultMaxBudget
	if v := jNum(data, "max_tokens"); v > 0 {
		maxTokens = int(v)
	}
	targetBudget := maxTokens - 1

	thinking := jMap(data, "thinking")

	if data["thinking"] == nil || (thinking != nil && jStr(thinking, "type") == "disabled") {
		data["thinking"] = map[string]any{
			"type":          "enabled",
			"budget_tokens": float64(targetBudget),
		}
		const thinkingBeta = "interleaved-thinking-2025-05-14"
		betas := jSlice(data, "anthropic_beta")
		if !sliceHas(betas, thinkingBeta) {
			betas = append(betas, thinkingBeta)
			data["anthropic_beta"] = betas
		}
		return true, fmt.Sprintf("injected budget=%d", targetBudget)
	}

	if thinking != nil {
		oldBudget := int(jNum(thinking, "budget_tokens"))
		if oldBudget < targetBudget {
			thinking["budget_tokens"] = float64(targetBudget)
			return true, fmt.Sprintf("upgraded %d->%d", oldBudget, targetBudget)
		}
		return false, fmt.Sprintf("already max budget=%d", oldBudget)
	}

	return false, "skip (unexpected format)"
}

// --- Cache Control Injection ---

func collectCacheBlocks(data map[string]any) []map[string]any {
	var blocks []map[string]any
	collect := func(items []any) {
		for _, item := range items {
			if m, ok := item.(map[string]any); ok {
				if _, has := m["cache_control"]; has {
					blocks = append(blocks, m)
				}
			}
		}
	}

	collect(jSlice(data, "tools"))
	collect(jSlice(data, "system"))

	for _, msg := range jSlice(data, "messages") {
		if m, ok := msg.(map[string]any); ok {
			collect(jSlice(m, "content"))
		}
	}
	return blocks
}

func newMarker() map[string]any {
	m := map[string]any{"type": "ephemeral"}
	if cacheTTL != "5m" {
		m["ttl"] = cacheTTL
	}
	return m
}

func injectCacheControl(data map[string]any) (int, string) {
	blocks := collectCacheBlocks(data)
	existing := len(blocks)

	upgraded := 0
	if cacheTTL != "5m" {
		for _, block := range blocks {
			if cc, ok := block["cache_control"].(map[string]any); ok {
				cc["ttl"] = cacheTTL
				upgraded++
			}
		}
	}

	budget := maxCacheBreakpoints - existing
	added := 0
	var parts []string

	if budget <= 0 {
		if upgraded > 0 {
			return 0, fmt.Sprintf("ttl-upgrade(%d->%s,existing=%d)", upgraded, cacheTTL, existing)
		}
		return 0, fmt.Sprintf("no-op(existing=%d)", existing)
	}

	// 1. Tools -- last tool
	if tools := jSlice(data, "tools"); len(tools) > 0 && added < budget {
		if last, ok := tools[len(tools)-1].(map[string]any); ok {
			if _, has := last["cache_control"]; !has {
				last["cache_control"] = newMarker()
				added++
				parts = append(parts, "tools")
			}
		}
	}

	// 2. System prompt
	if added < budget {
		if sysStr, ok := data["system"].(string); ok && sysStr != "" {
			data["system"] = []any{map[string]any{
				"type": "text", "text": sysStr, "cache_control": newMarker(),
			}}
			added++
			parts = append(parts, "system")
		} else if sys := jSlice(data, "system"); len(sys) > 0 {
			if last, ok := sys[len(sys)-1].(map[string]any); ok {
				if _, has := last["cache_control"]; !has {
					last["cache_control"] = newMarker()
					added++
					parts = append(parts, "system")
				}
			}
		}
	}

	// 3. Last assistant turn's last non-thinking block
	if added < budget {
		msgs := jSlice(data, "messages")
		for i := len(msgs) - 1; i >= 0; i-- {
			msg, ok := msgs[i].(map[string]any)
			if !ok || jStr(msg, "role") != "assistant" {
				continue
			}
			if cs, ok := msg["content"].(string); ok {
				msg["content"] = []any{map[string]any{
					"type": "text", "text": cs, "cache_control": newMarker(),
				}}
				added++
				parts = append(parts, "msgs")
			} else if content := jSlice(msg, "content"); len(content) > 0 {
				for j := len(content) - 1; j >= 0; j-- {
					block, ok := content[j].(map[string]any)
					if !ok {
						continue
					}
					typ := jStr(block, "type")
					if typ == "thinking" || typ == "redacted_thinking" {
						continue
					}
					if _, has := block["cache_control"]; !has {
						block["cache_control"] = newMarker()
						added++
						parts = append(parts, "msgs")
					}
					break
				}
			}
			break
		}
	}

	upg := ""
	if upgraded > 0 {
		upg = fmt.Sprintf(",upg=%d", upgraded)
	}
	if added > 0 {
		return added, fmt.Sprintf("%dbp(%s,%s,pre=%d%s)", added, strings.Join(parts, "+"), cacheTTL, existing, upg)
	}
	return 0, fmt.Sprintf("no-op(existing=%d%s)", existing, upg)
}

// --- EventStream Cache Metrics ---

var (
	b64MsgStart = []byte("eyJ0eXBlIjoibWVzc2FnZV9zdGFydC")
	b64MsgDelta = []byte("eyJ0eXBlIjoibWVzc2FnZV9kZWx0YS")
)

type cacheMetrics struct {
	read, write, input int
}

func extractCacheMetrics(chunk []byte, reqID int64) *cacheMetrics {
	if !bytes.Contains(chunk, b64MsgStart) && !bytes.Contains(chunk, b64MsgDelta) {
		return nil
	}

	buf := chunk
	for len(buf) >= 16 {
		totalLen := int(binary.BigEndian.Uint32(buf[0:4]))
		if totalLen < 16 || totalLen > len(buf) {
			break
		}
		headersLen := int(binary.BigEndian.Uint32(buf[4:8]))
		payloadStart := 12 + headersLen
		payloadEnd := totalLen - 4

		if payloadStart < payloadEnd && payloadEnd <= totalLen {
			payload := bytes.TrimSpace(buf[payloadStart:payloadEnd])
			if len(payload) > 0 && payload[0] == '{' {
				var wrapper map[string]any
				if json.Unmarshal(payload, &wrapper) == nil {
					obj := wrapper
					if b64, ok := wrapper["bytes"].(string); ok {
						if decoded, err := base64.StdEncoding.DecodeString(b64); err == nil {
							var inner map[string]any
							if json.Unmarshal(decoded, &inner) == nil {
								obj = inner
							}
						}
					}

					evtType := jStr(obj, "type")
					var usage map[string]any
					if evtType == "message_start" {
						if msg := jMap(obj, "message"); msg != nil {
							usage = jMap(msg, "usage")
						}
					} else if evtType == "message_delta" {
						usage = jMap(obj, "usage")
					}

					if usage != nil {
						cr := int(jNum(usage, "cache_read_input_tokens"))
						cw := int(jNum(usage, "cache_creation_input_tokens"))
						inp := int(jNum(usage, "input_tokens"))

						ttlInfo := ""
						if cc := jMap(usage, "cache_creation"); cc != nil {
							t5 := int(jNum(cc, "ephemeral_5m_input_tokens"))
							t1h := int(jNum(cc, "ephemeral_1h_input_tokens"))
							if t5 > 0 || t1h > 0 {
								ttlInfo = fmt.Sprintf(" [5m=%d,1h=%d]", t5, t1h)
							}
						}
						log.Printf("[#%d] cache-extract: %s cr=%d cw=%d inp=%d%s",
							reqID, evtType, cr, cw, inp, ttlInfo)
						if cr > 0 || cw > 0 {
							return &cacheMetrics{cr, cw, inp}
						}
					}
				}
			}
		}
		buf = buf[totalLen:]
	}
	return nil
}

// --- Proxy Handler ---

func proxyHandler(w http.ResponseWriter, r *http.Request) {
	reqID := reqCounter.Add(1)
	requestCount.Add(1)

	rawPath := r.URL.RawPath
	if rawPath == "" {
		rawPath = r.URL.Path
	}
	query := r.URL.RawQuery

	targetHost, resolvedPath, modelID := resolveTarget(rawPath)

	targetURL := "https://" + targetHost + resolvedPath
	if query != "" {
		targetURL += "?" + query
	}

	const maxBody = 50 << 20 // 50MB
	body, err := io.ReadAll(io.LimitReader(r.Body, maxBody+1))
	if err != nil {
		http.Error(w, `{"error":"read_body_failed"}`, http.StatusBadGateway)
		return
	}
	if len(body) > maxBody {
		http.Error(w, `{"error":"body_too_large"}`, http.StatusRequestEntityTooLarge)
		return
	}

	decodedPath, _ := url.PathUnescape(resolvedPath)
	if modelID == "" {
		if idx := strings.Index(decodedPath, "/model/"); idx >= 0 {
			rest := decodedPath[idx+len("/model/"):]
			modelID = strings.SplitN(rest, "/", 2)[0]
		}
	}

	isInvoke := strings.Contains(decodedPath, "/invoke")
	modelLower := strings.ToLower(modelID)
	thinkingAction := "passthrough"
	cacheAction := "off"

	if len(body) > 0 && isInvoke {
		var data map[string]any
		if err := json.Unmarshal(body, &data); err == nil {
			modified := false

			// Strip defer_loading from tools
			if tools := jSlice(data, "tools"); tools != nil {
				for _, t := range tools {
					if tool, ok := t.(map[string]any); ok {
						if _, has := tool["defer_loading"]; has {
							delete(tool, "defer_loading")
							modified = true
						}
					}
				}
			}

			// Thinking/effort modification
			thinkMod, action := modifyThinking(data, modelLower)
			thinkingAction = action
			if thinkMod {
				modified = true
				thinkingModCnt.Add(1)
			}

			// Prompt caching injection
			if cacheEnabled {
				if _, ok := data["messages"]; ok {
					cacheAdded, action := injectCacheControl(data)
					cacheAction = action
					if cacheAdded > 0 || strings.Contains(action, "ttl-upgrade") {
						modified = true
					}
					if cacheAdded > 0 {
						cacheInjCnt.Add(1)
					}
				}
			}

			if modified {
				body, _ = json.Marshal(data)
			}

			log.Printf("[#%d] -> %s model=%s | thinking: %s | cache: %s | body=%dB -> %s",
				reqID, r.Method, strOr(modelID, "n/a"), thinkingAction, cacheAction, len(body), targetHost)
		} else {
			log.Printf("[#%d] body parse error: %v", reqID, err)
			log.Printf("[#%d] -> %s %s (passthrough) -> %s", reqID, r.Method, decodedPath, targetHost)
		}
	} else {
		log.Printf("[#%d] -> %s %s -> %s", reqID, r.Method, decodedPath, targetHost)
	}

	// Build upstream request
	req, err := http.NewRequestWithContext(r.Context(), r.Method, targetURL, bytes.NewReader(body))
	if err != nil {
		http.Error(w, `{"error":"build_request_failed"}`, http.StatusBadGateway)
		return
	}

	ct := r.Header.Get("Content-Type")
	if ct == "" {
		ct = "application/json"
	}
	req.Header.Set("Content-Type", ct)

	if bearerToken != "" {
		req.Header.Set("Authorization", "Bearer "+bearerToken)
	} else {
		// SigV4 signing
		creds, err := awsCredProvider.Retrieve(r.Context())
		if err != nil {
			log.Printf("[#%d] credential error: %v", reqID, err)
			http.Error(w, `{"error":"credentials_failed"}`, http.StatusBadGateway)
			return
		}
		hash := sha256.Sum256(body)
		payloadHash := hex.EncodeToString(hash[:])
		if err := awsSigner.SignHTTP(r.Context(), creds, req, payloadHash, "bedrock", targetRegion, time.Now()); err != nil {
			log.Printf("[#%d] signing error: %v", reqID, err)
			http.Error(w, `{"error":"signing_failed"}`, http.StatusBadGateway)
			return
		}
	}

	isStreaming := strings.Contains(resolvedPath, "response-stream")
	t0 := time.Now()

	resp, err := httpClient.Do(req)
	if err != nil {
		log.Printf("[#%d] upstream error after %.1fs: %v", reqID, time.Since(t0).Seconds(), err)
		http.Error(w, `{"error":"upstream_error"}`, http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		errBody, err := io.ReadAll(resp.Body)
		if err != nil {
			log.Printf("[#%d] <- %d %s error reading body: %v", reqID, resp.StatusCode, resp.Proto, err)
			http.Error(w, `{"error":"upstream_read_failed"}`, http.StatusBadGateway)
			return
		}
		preview := string(errBody)
		if len(preview) > 500 {
			preview = preview[:500]
		}
		log.Printf("[#%d] <- %d %s in %.1fs | %s",
			reqID, resp.StatusCode, resp.Proto, time.Since(t0).Seconds(), preview)
		copyResponseHeaders(w, resp)
		w.WriteHeader(resp.StatusCode)
		w.Write(errBody)
		return
	}

	if isStreaming {
		streamResponse(w, resp, reqID, t0)
		return
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("[#%d] <- %d %s error reading body: %v", reqID, resp.StatusCode, resp.Proto, err)
		http.Error(w, `{"error":"upstream_read_failed"}`, http.StatusBadGateway)
		return
	}
	log.Printf("[#%d] <- %d %s %dB in %.1fs",
		reqID, resp.StatusCode, resp.Proto, len(respBody), time.Since(t0).Seconds())
	copyResponseHeaders(w, resp)
	w.WriteHeader(resp.StatusCode)
	w.Write(respBody)
}

// --- Streaming ---

var skipHeaders = map[string]bool{
	"Transfer-Encoding": true,
	"Content-Encoding":  true,
	"Connection":        true,
}

func copyResponseHeaders(w http.ResponseWriter, resp *http.Response) {
	for k, vs := range resp.Header {
		if skipHeaders[k] {
			continue
		}
		for _, v := range vs {
			w.Header().Add(k, v)
		}
	}
}

func streamResponse(w http.ResponseWriter, resp *http.Response, reqID int64, t0 time.Time) {
	flusher, canFlush := w.(http.Flusher)

	copyResponseHeaders(w, resp)
	w.WriteHeader(resp.StatusCode)

	totalBytes := 0
	cacheLogged := false
	buf := make([]byte, 32*1024)

	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := w.Write(buf[:n]); werr != nil {
				log.Printf("[#%d] downstream write error: %v", reqID, werr)
				break
			}
			if canFlush {
				flusher.Flush()
			}
			totalBytes += n

			if !cacheLogged {
				if m := extractCacheMetrics(buf[:n], reqID); m != nil {
					cacheLogged = true
					cacheReadTotal.Add(int64(m.read))
					cacheWriteTotal.Add(int64(m.write))
					log.Printf("[#%d] cache: read=%d write=%d uncached=%d",
						reqID, m.read, m.write, m.input)
				}
			}
		}
		if err != nil {
			break
		}
	}

	log.Printf("[#%d] <- %d %s streamed %dB in %.1fs",
		reqID, resp.StatusCode, resp.Proto, totalBytes, time.Since(t0).Seconds())
}

// --- Health ---

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"status":             "ok",
		"protocol":           "h2 (upstream)",
		"region":             targetRegion,
		"runtime":            runtimeHost,
		"control":            controlHost,
		"requests_served":    requestCount.Load(),
		"thinking_modified":  thinkingModCnt.Load(),
		"cache_enabled":      cacheEnabled,
		"cache_ttl":          cacheTTL,
		"cache_injected":     cacheInjCnt.Load(),
		"cache_read_tokens":  cacheReadTotal.Load(),
		"cache_write_tokens": cacheWriteTotal.Load(),
	})
}

// --- Main ---

func main() {
	if bearerToken == "" {
		cfg, err := config.LoadDefaultConfig(context.Background(),
			config.WithRegion(targetRegion))
		if err != nil {
			log.Fatalf("AWS config load failed: %v", err)
		}
		awsCredProvider = cfg.Credentials
		awsSigner = v4.NewSigner()
	}

	authMode := "bearer"
	if bearerToken == "" {
		authMode = "sigv4"
	}

	log.Printf("Bedrock Effort Max Proxy (Go/h2) starting on %s", listenAddr)
	log.Printf("Runtime: %s (h2 via ALPN)", runtimeHost)
	log.Printf("Control: %s", controlHost)
	log.Printf("Auth: %s", authMode)
	log.Printf("Model remaps: %d entries", len(modelMap))
	log.Printf("Adaptive models (effort=max): %v", adaptiveModels)
	log.Printf("Legacy models: budget_tokens maximized")
	log.Printf("Skip: %v", thinkingSkipModels)
	log.Printf("Cache: enabled=%v, ttl=%s", cacheEnabled, cacheTTL)

	mux := http.NewServeMux()
	mux.HandleFunc("/health", healthHandler)
	mux.HandleFunc("/", proxyHandler)

	srv := &http.Server{
		Addr:           listenAddr,
		Handler:        mux,
		ReadTimeout:    30 * time.Second,
		WriteTimeout:   0, // disabled for streaming
		MaxHeaderBytes: 1 << 20,
	}

	// Graceful shutdown on SIGINT/SIGTERM
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		sig := <-sigCh
		log.Printf("Received %v, shutting down...", sig)
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		srv.Shutdown(ctx)
	}()

	if err := srv.ListenAndServe(); err != http.ErrServerClosed {
		log.Fatalf("Server failed: %v", err)
	}
	log.Printf("Server stopped. Served %d requests.", requestCount.Load())
}

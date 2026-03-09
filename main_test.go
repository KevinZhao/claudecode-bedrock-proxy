package main

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// --- resolveTarget ---

func TestResolveTarget_RuntimeDefault(t *testing.T) {
	host, path, mid := resolveTarget("/model/some-model/invoke")
	if host != runtimeHost {
		t.Errorf("host = %q, want %q", host, runtimeHost)
	}
	if path != "/model/some-model/invoke" {
		t.Errorf("path = %q", path)
	}
	if mid != "some-model" {
		t.Errorf("modelID = %q", mid)
	}
}

func TestResolveTarget_ControlPlane(t *testing.T) {
	for _, prefix := range []string{"/inference-profiles", "/foundation-models", "/guardrails"} {
		host, _, mid := resolveTarget(prefix + "/foo")
		if host != controlHost {
			t.Errorf("prefix %s: host = %q, want %q", prefix, host, controlHost)
		}
		if mid != "" {
			t.Errorf("prefix %s: modelID should be empty, got %q", prefix, mid)
		}
	}
}

func TestResolveTarget_ModelRemap(t *testing.T) {
	tests := []struct {
		input   string
		wantID  string
		wantMid string
	}{
		{"/model/claude-opus-4-6/invoke", "global.anthropic.claude-opus-4-6-v1", "global.anthropic.claude-opus-4-6-v1"},
		{"/model/claude-sonnet-4-6/invoke-with-response-stream", "global.anthropic.claude-sonnet-4-6", "global.anthropic.claude-sonnet-4-6"},
		{"/model/claude-haiku-4-5/invoke", "global.anthropic.claude-haiku-4-5-20251001-v1:0", "global.anthropic.claude-haiku-4-5-20251001-v1:0"},
	}
	for _, tt := range tests {
		_, path, mid := resolveTarget(tt.input)
		if mid != tt.wantMid {
			t.Errorf("resolveTarget(%q) modelID = %q, want %q", tt.input, mid, tt.wantMid)
		}
		if !strings.Contains(path, tt.wantID) {
			t.Errorf("resolveTarget(%q) path = %q, should contain %q", tt.input, path, tt.wantID)
		}
	}
}

func TestResolveTarget_UnknownModel(t *testing.T) {
	host, path, mid := resolveTarget("/model/unknown-model/invoke")
	if host != runtimeHost {
		t.Errorf("host = %q", host)
	}
	if path != "/model/unknown-model/invoke" {
		t.Errorf("path modified for unknown model: %q", path)
	}
	if mid != "unknown-model" {
		t.Errorf("modelID = %q", mid)
	}
}

// --- modifyThinking ---

func TestModifyThinking_SkipHaiku(t *testing.T) {
	data := map[string]any{"messages": []any{}}
	modified, desc := modifyThinking(data, "claude-haiku-4-5")
	if modified {
		t.Error("should not modify haiku")
	}
	if !strings.Contains(desc, "haiku") {
		t.Errorf("desc = %q", desc)
	}
}

func TestModifyThinking_NoMessages(t *testing.T) {
	data := map[string]any{}
	modified, desc := modifyThinking(data, "opus-4-6")
	if modified {
		t.Error("should passthrough without messages")
	}
	if desc != "passthrough" {
		t.Errorf("desc = %q", desc)
	}
}

func TestModifyAdaptiveThinking_Opus46(t *testing.T) {
	data := map[string]any{"messages": []any{}}
	modified, _ := modifyThinking(data, "global.anthropic.claude-opus-4-6-v1")
	if !modified {
		t.Error("should modify opus-4-6")
	}
	thinking := jMap(data, "thinking")
	if thinking == nil || jStr(thinking, "type") != "adaptive" {
		t.Errorf("thinking = %v", data["thinking"])
	}
	oc := jMap(data, "output_config")
	if oc == nil || jStr(oc, "effort") != "max" {
		t.Errorf("output_config = %v", data["output_config"])
	}
	betas := jSlice(data, "anthropic_beta")
	if !sliceHas(betas, "context-1m-2025-08-07") {
		t.Errorf("anthropic_beta = %v", betas)
	}
}

func TestModifyAdaptiveThinking_Sonnet46(t *testing.T) {
	data := map[string]any{"messages": []any{}}
	modified, _ := modifyThinking(data, "global.anthropic.claude-sonnet-4-6")
	if !modified {
		t.Error("should modify sonnet-4-6")
	}
	if jStr(jMap(data, "thinking"), "type") != "adaptive" {
		t.Errorf("thinking = %v", data["thinking"])
	}
}

func TestModifyAdaptiveThinking_AlreadySet(t *testing.T) {
	data := map[string]any{
		"messages":       []any{},
		"thinking":       map[string]any{"type": "adaptive"},
		"output_config":  map[string]any{"effort": "max"},
		"anthropic_beta": []any{"context-1m-2025-08-07"},
	}
	modified, desc := modifyThinking(data, "opus-4-6")
	if modified {
		t.Error("should not modify when already set")
	}
	if !strings.Contains(desc, "already") {
		t.Errorf("desc = %q", desc)
	}
}

func TestModifyAdaptiveThinking_EffortUpgrade(t *testing.T) {
	data := map[string]any{
		"messages":      []any{},
		"thinking":      map[string]any{"type": "adaptive"},
		"output_config": map[string]any{"effort": "low"},
	}
	modified, _ := modifyThinking(data, "sonnet-4-6")
	if !modified {
		t.Error("should upgrade effort")
	}
	if jStr(jMap(data, "output_config"), "effort") != "max" {
		t.Errorf("effort = %v", jMap(data, "output_config"))
	}
}

func TestModifyLegacyThinking_Sonnet4(t *testing.T) {
	data := map[string]any{
		"messages":   []any{},
		"max_tokens": float64(8192),
	}
	modified, _ := modifyThinking(data, "global.anthropic.claude-sonnet-4-20250514-v1:0")
	if !modified {
		t.Error("should modify legacy model")
	}
	thinking := jMap(data, "thinking")
	if thinking == nil {
		t.Fatal("thinking is nil")
	}
	if jStr(thinking, "type") != "enabled" {
		t.Errorf("type = %q", jStr(thinking, "type"))
	}
	if int(jNum(thinking, "budget_tokens")) != 8191 {
		t.Errorf("budget_tokens = %v", thinking["budget_tokens"])
	}
}

func TestModifyLegacyThinking_DefaultBudget(t *testing.T) {
	data := map[string]any{"messages": []any{}}
	modified, desc := modifyThinking(data, "global.anthropic.claude-sonnet-4-20250514-v1:0")
	if !modified {
		t.Error("should inject default budget")
	}
	budget := int(jNum(jMap(data, "thinking"), "budget_tokens"))
	if budget != defaultMaxBudget-1 {
		t.Errorf("budget = %d, want %d", budget, defaultMaxBudget-1)
	}
	if !strings.Contains(desc, "injected") {
		t.Errorf("desc = %q", desc)
	}
}

func TestModifyLegacyThinking_UpgradeBudget(t *testing.T) {
	data := map[string]any{
		"messages":   []any{},
		"max_tokens": float64(8192),
		"thinking":   map[string]any{"type": "enabled", "budget_tokens": float64(1000)},
	}
	modified, desc := modifyThinking(data, "claude-sonnet-4-20250514")
	if !modified {
		t.Error("should upgrade budget")
	}
	if !strings.Contains(desc, "upgraded") {
		t.Errorf("desc = %q", desc)
	}
}

// --- injectCacheControl ---

func TestInjectCacheControl_LastTool(t *testing.T) {
	tools := []any{
		map[string]any{"name": "t1"},
		map[string]any{"name": "t2"},
	}
	data := map[string]any{
		"tools":    tools,
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
	}
	added, _ := injectCacheControl(data)
	if added < 1 {
		t.Error("should inject at least on tools")
	}
	last := tools[len(tools)-1].(map[string]any)
	if _, has := last["cache_control"]; !has {
		t.Error("last tool missing cache_control")
	}
}

func TestInjectCacheControl_SystemString(t *testing.T) {
	data := map[string]any{
		"system":   "You are helpful.",
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
	}
	added, _ := injectCacheControl(data)
	if added < 1 {
		t.Error("should inject on system")
	}
	sys, ok := data["system"].([]any)
	if !ok {
		t.Fatal("system should be converted to array")
	}
	block := sys[0].(map[string]any)
	if _, has := block["cache_control"]; !has {
		t.Error("system block missing cache_control")
	}
}

func TestInjectCacheControl_SystemArray(t *testing.T) {
	sys := []any{
		map[string]any{"type": "text", "text": "Block 1"},
		map[string]any{"type": "text", "text": "Block 2"},
	}
	data := map[string]any{
		"system":   sys,
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
	}
	injectCacheControl(data)
	last := sys[len(sys)-1].(map[string]any)
	if _, has := last["cache_control"]; !has {
		t.Error("last system block missing cache_control")
	}
}

func TestInjectCacheControl_LastAssistantMessage(t *testing.T) {
	msgs := []any{
		map[string]any{"role": "user", "content": "hello"},
		map[string]any{"role": "assistant", "content": []any{
			map[string]any{"type": "text", "text": "response"},
		}},
		map[string]any{"role": "user", "content": "follow up"},
	}
	data := map[string]any{"messages": msgs}
	added, _ := injectCacheControl(data)
	if added < 1 {
		t.Error("should inject on assistant message")
	}
	content := msgs[1].(map[string]any)["content"].([]any)
	block := content[0].(map[string]any)
	if _, has := block["cache_control"]; !has {
		t.Error("assistant content block missing cache_control")
	}
}

func TestInjectCacheControl_SkipsThinkingBlocks(t *testing.T) {
	msgs := []any{
		map[string]any{"role": "assistant", "content": []any{
			map[string]any{"type": "thinking", "thinking": "..."},
			map[string]any{"type": "text", "text": "answer"},
			map[string]any{"type": "thinking", "thinking": "more"},
		}},
	}
	data := map[string]any{"messages": msgs}
	injectCacheControl(data)
	content := msgs[0].(map[string]any)["content"].([]any)
	// Should inject on "text" block, not "thinking"
	textBlock := content[1].(map[string]any)
	if _, has := textBlock["cache_control"]; !has {
		t.Error("text block should get cache_control")
	}
	thinkBlock := content[2].(map[string]any)
	if _, has := thinkBlock["cache_control"]; has {
		t.Error("thinking block should NOT get cache_control")
	}
}

func TestInjectCacheControl_BudgetLimit(t *testing.T) {
	data := map[string]any{
		"tools":  []any{map[string]any{"name": "t"}},
		"system": []any{map[string]any{"type": "text", "text": "sys"}},
		"messages": []any{
			map[string]any{"role": "user", "content": "hi"},
			map[string]any{"role": "assistant", "content": []any{
				map[string]any{"type": "text", "text": "resp"},
			}},
		},
	}
	injectCacheControl(data)
	total := len(collectCacheBlocks(data))
	if total > maxCacheBreakpoints {
		t.Errorf("total cache blocks = %d, max = %d", total, maxCacheBreakpoints)
	}
}

func TestInjectCacheControl_NoOpWhenExhausted(t *testing.T) {
	data := map[string]any{
		"tools": []any{
			map[string]any{"name": "t1", "cache_control": map[string]any{"type": "ephemeral"}},
			map[string]any{"name": "t2", "cache_control": map[string]any{"type": "ephemeral"}},
		},
		"system": []any{
			map[string]any{"type": "text", "text": "s1", "cache_control": map[string]any{"type": "ephemeral"}},
			map[string]any{"type": "text", "text": "s2", "cache_control": map[string]any{"type": "ephemeral"}},
		},
		"messages": []any{},
	}
	added, desc := injectCacheControl(data)
	if added != 0 {
		t.Errorf("added = %d, want 0", added)
	}
	if !strings.Contains(desc, "no-op") && !strings.Contains(desc, "ttl-upgrade") {
		t.Errorf("desc = %q", desc)
	}
}

func TestInjectCacheControl_EmptyTools(t *testing.T) {
	data := map[string]any{
		"tools":    []any{},
		"system":   "sys",
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
	}
	_, desc := injectCacheControl(data)
	if strings.Contains(desc, "tools") {
		t.Errorf("should not inject on empty tools, desc = %q", desc)
	}
}

// --- defer_loading stripping ---

func TestStripDeferLoading(t *testing.T) {
	tools := []any{
		map[string]any{"name": "t1"},
		map[string]any{"name": "t2", "defer_loading": true},
	}
	for _, t := range tools {
		if tool, ok := t.(map[string]any); ok {
			delete(tool, "defer_loading")
		}
	}
	t2 := tools[1].(map[string]any)
	if _, has := t2["defer_loading"]; has {
		t.Error("defer_loading should be stripped")
	}
}

// --- extractCacheMetrics ---

func buildEventStreamFrame(payload []byte) []byte {
	headersLen := 0
	totalLen := 12 + headersLen + len(payload) + 4
	frame := make([]byte, totalLen)
	binary.BigEndian.PutUint32(frame[0:4], uint32(totalLen))
	binary.BigEndian.PutUint32(frame[4:8], uint32(headersLen))
	// prelude CRC at [8:12] — zero for testing
	copy(frame[12:], payload)
	// message CRC at end — zero for testing
	return frame
}

func TestExtractCacheMetrics_MessageStart(t *testing.T) {
	// Key order matters: "type" must come first to match base64 prefix filter.
	// json.Marshal sorts alphabetically ("message" before "type"), so use raw JSON.
	innerJSON := []byte(`{"type":"message_start","message":{"usage":{"input_tokens":100,"cache_read_input_tokens":5000,"cache_creation_input_tokens":200}}}`)
	b64 := base64.StdEncoding.EncodeToString(innerJSON)
	wrapperJSON := []byte(`{"bytes":"` + b64 + `"}`)

	frame := buildEventStreamFrame(wrapperJSON)
	m := extractCacheMetrics(frame, 1)
	if m == nil {
		t.Fatal("should extract metrics")
	}
	if m.read != 5000 {
		t.Errorf("read = %d, want 5000", m.read)
	}
	if m.write != 200 {
		t.Errorf("write = %d, want 200", m.write)
	}
	if m.input != 100 {
		t.Errorf("input = %d, want 100", m.input)
	}
}

func TestExtractCacheMetrics_NoCacheTokens(t *testing.T) {
	inner := map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"usage": map[string]any{
				"input_tokens":                float64(100),
				"cache_read_input_tokens":     float64(0),
				"cache_creation_input_tokens": float64(0),
			},
		},
	}
	innerJSON, _ := json.Marshal(inner)
	b64 := base64.StdEncoding.EncodeToString(innerJSON)
	wrapper := map[string]any{"bytes": b64}
	wrapperJSON, _ := json.Marshal(wrapper)

	frame := buildEventStreamFrame(wrapperJSON)
	m := extractCacheMetrics(frame, 1)
	if m != nil {
		t.Error("should return nil when cr=0 and cw=0")
	}
}

func TestExtractCacheMetrics_NoMatch(t *testing.T) {
	chunk := []byte("random data without base64 markers")
	m := extractCacheMetrics(chunk, 1)
	if m != nil {
		t.Error("should return nil for non-EventStream data")
	}
}

func TestExtractCacheMetrics_MessageDelta(t *testing.T) {
	inner := map[string]any{
		"type": "message_delta",
		"usage": map[string]any{
			"cache_read_input_tokens":     float64(3000),
			"cache_creation_input_tokens": float64(100),
		},
	}
	innerJSON, _ := json.Marshal(inner)
	b64 := base64.StdEncoding.EncodeToString(innerJSON)
	wrapper := map[string]any{"bytes": b64}
	wrapperJSON, _ := json.Marshal(wrapper)

	frame := buildEventStreamFrame(wrapperJSON)
	m := extractCacheMetrics(frame, 1)
	if m == nil {
		t.Fatal("should extract delta metrics")
	}
	if m.read != 3000 || m.write != 100 {
		t.Errorf("read=%d write=%d", m.read, m.write)
	}
}

// --- HTTP handler integration ---

func TestHealthEndpoint(t *testing.T) {
	req := httptest.NewRequest("GET", "/health", nil)
	w := httptest.NewRecorder()
	healthHandler(w, req)

	if w.Code != 200 {
		t.Errorf("status = %d", w.Code)
	}
	ct := w.Header().Get("Content-Type")
	if !strings.Contains(ct, "application/json") {
		t.Errorf("content-type = %q", ct)
	}
	var body map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if body["status"] != "ok" {
		t.Errorf("status = %v", body["status"])
	}
}

func TestProxyHandler_BodyTooLarge(t *testing.T) {
	big := strings.NewReader(strings.Repeat("x", 50<<20+1))
	req := httptest.NewRequest("POST", "/model/test/invoke", big)
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	proxyHandler(w, req)

	if w.Code != http.StatusRequestEntityTooLarge {
		t.Errorf("status = %d, want 413", w.Code)
	}
}

func TestCopyResponseHeaders(t *testing.T) {
	resp := &http.Response{
		Header: http.Header{
			"Content-Type":      {"application/json"},
			"X-Amzn-Requestid":  {"abc-123"},
			"Transfer-Encoding": {"chunked"},
			"Connection":        {"keep-alive"},
		},
	}
	w := httptest.NewRecorder()
	copyResponseHeaders(w, resp)

	if w.Header().Get("Content-Type") != "application/json" {
		t.Error("Content-Type not copied")
	}
	if w.Header().Get("X-Amzn-Requestid") != "abc-123" {
		t.Error("X-Amzn-Requestid not copied")
	}
	if w.Header().Get("Transfer-Encoding") != "" {
		t.Error("Transfer-Encoding should be skipped")
	}
	if w.Header().Get("Connection") != "" {
		t.Error("Connection should be skipped")
	}
}

func TestStreamResponse_WriteError(t *testing.T) {
	body := io.NopCloser(strings.NewReader("test stream data"))
	resp := &http.Response{
		StatusCode: 200,
		Proto:      "HTTP/2.0",
		Header:     http.Header{"Content-Type": {"application/octet-stream"}},
		Body:       body,
	}

	w := httptest.NewRecorder()
	streamResponse(w, resp, 99, timeNow())
	if w.Code != 200 {
		t.Errorf("status = %d", w.Code)
	}
	if w.Body.Len() == 0 {
		t.Error("body should not be empty")
	}
}

func timeNow() time.Time { return time.Now() }

// --- JSON helpers ---

func TestJHelpers(t *testing.T) {
	data := map[string]any{
		"str":   "hello",
		"num":   float64(42),
		"map":   map[string]any{"nested": true},
		"slice": []any{"a", "b"},
	}
	if jStr(data, "str") != "hello" {
		t.Error("jStr")
	}
	if jStr(data, "missing") != "" {
		t.Error("jStr missing")
	}
	if jNum(data, "num") != 42 {
		t.Error("jNum")
	}
	if jNum(data, "missing") != 0 {
		t.Error("jNum missing")
	}
	if jMap(data, "map") == nil {
		t.Error("jMap")
	}
	if jMap(data, "missing") != nil {
		t.Error("jMap missing")
	}
	if jSlice(data, "slice") == nil {
		t.Error("jSlice")
	}
	if jSlice(data, "missing") != nil {
		t.Error("jSlice missing")
	}
}

# Bedrock Effort Max Proxy

本地反向代理，部署在 Claude Code CLI 与 AWS Bedrock 之间，用于弥补 Claude Code 在 Bedrock 场景下的几个已知不足。

## 解决的问题

### 1. Prompt Caching TTL 不可配置（主 CLI）

**问题：** Claude Code 主进程（Opus）会自动注入 prompt caching breakpoints，但 TTL 固定为 5m（默认值），无法配置为 1h。对于长时间开发会话，5 分钟后 cache 过期导致全量重新写入，浪费大量 input tokens 费用。

**Proxy 行为：** 将所有已有 `cache_control` 的 TTL 从 5m 升级为 1h。

**实际效果（来自日志）：**
```
# 5m TTL 过期后：61k tokens 全量重写 ≈ $1.14
[#11] cache: read=0      write=61149  ← cache miss, 全量重写

# 升级 1h TTL 后：后续请求全部命中
[#14] cache: read=61149   write=246   ← cache hit
```

### 2. Agent SDK 完全缺失 Prompt Caching（Agent tool Subagent）

**问题：** Claude Code 有两种多 agent 机制，prompt caching 支持不同：

| 机制 | 进程模型 | User-Agent | 自带 Prompt Caching |
|------|----------|------------|---------------------|
| **Agent tool** | 主进程内通过 `agent-sdk` 调用子进程 | `agent-sdk/0.1.76` | 无 |
| **Agent Teams**（tmux 模式） | 独立终端窗口，各自运行完整 `claude` CLI | `cli` | 有（同主 CLI） |

Agent tool 的 `agent-sdk`（截至 v0.1.76）**完全不注入** `cache_control` breakpoints，所有 subagent 请求的 tools 和 system prompt 每次都作为 uncached input 计费。Agent Teams 的 teammates 因为是独立 CLI 进程，自带 breakpoints，仅需 TTL 升级（见问题 1）。

**Proxy 行为：** 对无 breakpoints 的请求（`pre=0`），自动在 tools、system prompt、最后一条 assistant message 上注入 `cache_control`（最多 4 个断点，遵守 API 限制）。

**实际效果：**
```
# Agent tool subagent (agent-sdk)，proxy 注入后：
[#1]  cache: 2bp(tools+system,1h,pre=0)  → cr=7013  cw=0     ← 命中缓存
[#13] cache: 2bp(tools+system,1h,pre=0)  → cr=7013  cw=0     ← 持续命中

# Agent Teams teammate (cli)，仅 TTL 升级：
[#5]  cache: ttl-upgrade(4->1h,existing=4) → cr=60726 cw=95  ← 自带 caching
```

### 3. Thinking 配置未针对最新模型优化

**问题：** Claude Code 对 Opus 4.6 / Sonnet 4.6 仍使用旧版 `budget_tokens` thinking 模式或未设置 `effort=max`，无法充分利用新模型的 adaptive thinking 能力。

**Proxy 行为：**

| 模型 | 策略 |
|------|------|
| Opus 4.6 / Sonnet 4.6 | `thinking: adaptive` + `effort: max` + `context-1m` beta |
| 旧模型（Sonnet 4.5 等） | `budget_tokens` 最大化至 `max_tokens - 1` |
| Haiku | 跳过（不支持 thinking） |

### 4. 模型 ID 映射与版本锁定

**问题：** Claude Code 使用 Anthropic API 格式的 model ID（如 `claude-sonnet-4-5`），但 Bedrock 需要跨区域推理 profile ID。此外，无法在不更新 CC 配置的情况下将旧版模型请求路由到新版。

**Proxy 行为：** 自动映射，同时支持版本升级锁定：

```
claude-opus-4-6          → global.anthropic.claude-opus-4-6-v1
claude-sonnet-4-5        → global.anthropic.claude-sonnet-4-6      ← 自动升级
claude-sonnet-4-5-20250929 → global.anthropic.claude-sonnet-4-6
claude-haiku-4-5         → global.anthropic.claude-haiku-4-5-20251001-v1:0
```

## 架构

```
Claude Code CLI
      │
      ▼
  127.0.0.1:8888  (proxy)
      │
      ├─ 解析 & 修改 request body
      │   ├─ thinking/effort 配置
      │   ├─ cache_control 注入 / TTL 升级
      │   └─ model ID 映射
      │
      ▼
  AWS Bedrock (bedrock-runtime / bedrock)
```

## 使用

```bash
# 启动
./start.sh

# 停止
./stop.sh

# 健康检查
curl http://127.0.0.1:8888/health
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PROXY_PORT` | `8888` | 监听端口 |
| `AWS_REGION` | `ap-northeast-1` | Bedrock 区域 |
| `BEDROCK_BEARER_TOKEN` | - | 设置后使用 Bearer 认证，否则 SigV4 |
| `CACHE_ENABLED` | `1` | 启用 prompt caching 注入 |
| `CACHE_TTL` | `1h` | Cache TTL（`5m` 或 `1h`） |

## Claude Code 配置

参考 [`settings.json`](settings.json) 配置 Claude Code（`~/.claude/settings.json`），包含：

- Bedrock 模式启用 + proxy 指向
- 模型配置（主模型、各级别默认模型、`[1m]` context window）
- 性能调优选项
- SessionStart hook 自动启动 proxy

认证支持两种方式：**SigV4**（推荐，使用 `~/.aws/credentials` 或 Instance Profile）或 **Bearer Token**。

## 依赖

- Go 1.21+（默认，推荐）

### Python 版本（deprecated）

`proxy.py` 为早期 Python 实现，已废弃，不再维护。如需使用：

```bash
python3 proxy.py >> proxy.log 2>&1 &
```

依赖：Python 3.9+、`aiohttp`、`botocore`、`yarl`

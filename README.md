> # ⚠️ 本项目已归档（Archived / No Longer Maintained）
>
> **更新日期：2026-05-03**
>
> 本 proxy 最初用于弥补 Claude Code 在 Bedrock 场景下的一些配置传递缺陷（effort level、thinking mode、1h prompt cache TTL、1M context beta、subagent 配置继承等）。
>
> **随着 Claude Code 版本迭代，上述绝大多数功能 CC 本身已经原生支持**（包括主线程与 subagent 的配置一致性、1h cache TTL、effort=max、adaptive thinking、1M context 等）。因此本 proxy 的核心价值已被 CC 官方覆盖，**不再需要作为 CC 使用 Bedrock 的必备组件**。
>
> ## 建议
>
> - **Claude Code + Bedrock 用户：** 直接使用 CC 官方的 Bedrock 支持即可，无需部署本 proxy。相关配置参考 [Claude Code 官方文档](https://docs.claude.com/en/docs/claude-code/amazon-bedrock)。
> - **其他使用 Anthropic Messages API 格式的应用（Cursor、Continue、Cline 等）：** 如果仍需要在这些工具中访问 Bedrock 并获得 prompt caching 注入、model ID 映射等增强，本仓库代码可作为参考实现，但已不再维护，使用风险自负。
>
> ## 仓库状态
>
> - ❌ 不再接受 issue 与 PR
> - ❌ 不再发布新版本
> - ✅ 代码将保留作为历史参考
>
> 感谢所有使用过本项目的朋友。下面的原始文档保留以供参考。
>
> ---

# Bedrock Effort Max Proxy

本地反向代理，部署在 Claude Code CLI 与 AWS Bedrock 之间，用于弥补 Claude Code 在 Bedrock 场景下的已知不足。

> **CC v2.1.79 兼容性说明（2026-03-19）：** CC 主线程已原生支持 `effort: max` 和 1h prompt cache TTL，但 **subagent 请求仍存在多项配置传递缺陷**，proxy 对 subagent 仍是全方位有用的。详见下方"CC 原生能力 vs Proxy"对比表。

## CC 原生能力 vs Proxy

基于 CC v2.1.79 + Bedrock 的实测结果（通过 proxy 日志验证）：

| 功能 | CC 原生（主线程） | CC 原生（Subagent） | Proxy 补充 |
|------|-------------------|---------------------|------------|
| **Effort level** | `max` ✅ (`CLAUDE_CODE_EFFORT_LEVEL=max`) | 空 ❌ | 注入 `max` |
| **Thinking mode** | `adaptive` ✅ | `budget_tokens: 8192` ❌ | 升级为 `adaptive` |
| **1h Cache TTL** | `1h` ✅ (`ENABLE_PROMPT_CACHING_1H_BEDROCK=1`) | `5m` ❌ | 升级为 `1h` |
| **1M context beta** | 已包含 ✅ | 缺失 ❌ | 注入 `context-1m-2025-08-07` |
| **Model ID** | 正确 ✅ | 发送旧版 `claude-sonnet-4-5` ❌ | 映射为 `sonnet-4-6` |
| **Cache breakpoints** | 自带 3 个 ✅ | 自带 2 个 ✅ | 补充注入（如不足 4 个） |
| **SigV4 签名** | 需要 proxy | 需要 proxy | 统一处理 |

> **关键发现：** CC 的环境变量（`CLAUDE_CODE_EFFORT_LEVEL`、`ENABLE_PROMPT_CACHING_1H_BEDROCK`、`ANTHROPIC_DEFAULT_SONNET_MODEL`）只影响主线程，subagent 走内部硬编码路径。
> 已提交 issue：[#36243](https://github.com/anthropics/claude-code/issues/36243)（cache TTL）、[#36249](https://github.com/anthropics/claude-code/issues/36249)（全部 5 项）

## CC 原生配置（推荐同时启用）

在 `~/.claude/settings.json` 的 `env` 中添加：

```jsonc
"CLAUDE_CODE_EFFORT_LEVEL": "max",            // CC 原生 max effort（Opus 4.6 only）
"ENABLE_PROMPT_CACHING_1H_BEDROCK": "1"       // CC 原生 1h cache TTL（未文档化，仅主线程生效）
```

这两个设置可减少 proxy 对主线程请求的修改量，但 **不影响 subagent**（仍需 proxy）。

## Proxy 解决的问题

### 1. Subagent 配置传递缺陷（核心价值）

**问题：** CC 的 subagent（通过 Agent tool 调用的 Sonnet）不继承用户配置。5 项配置全部缺失：model ID、thinking mode、effort level、1M beta header、cache TTL。

**Proxy 行为：** 统一拦截所有请求（主线程 + subagent），确保配置一致性。

**实际效果（proxy 日志）：**
```
# 主线程（Opus）— CC 原生已正确，proxy 无需修改：
[#38] model=opus-4-6-v1 | thinking: already adaptive+max | cache: 1bp(tools,1h,pre=3,native=3)

# Subagent（Sonnet）— proxy 修复全部 5 项：
[#40] model=sonnet-4-6 | thinking: thinking->adaptive (was budget_tokens:8192); effort->max (was ); beta+=context-1m | cache: 1bp(tools,1h,pre=2,upg=2)
```

### 2. Prompt Caching 注入与 TTL 升级

**Proxy 行为：**
- 对已有 breakpoints 但 TTL 为 5m 的请求：升级为 1h（`upg=N`）
- 对无 breakpoints 的请求：注入最多 4 个断点 + 1h TTL
- 对已经是 1h 的请求：跳过（`native=N`）

### 3. Thinking 配置优化

| 模型 | Proxy 策略 |
|------|------|
| Opus 4.6 / Sonnet 4.6 | `thinking: adaptive` + `effort: max` + `context-1m` beta |
| 旧模型（Sonnet 4.5 等） | `budget_tokens` 最大化至 `max_tokens - 1` |
| Haiku | 跳过（不支持 thinking） |

### 4. 模型 ID 映射与版本锁定

CC subagent 发送旧版 model ID，proxy 自动映射：

```
claude-opus-4-6            → global.anthropic.claude-opus-4-6-v1
claude-sonnet-4-5          → global.anthropic.claude-sonnet-4-6      ← 自动升级
claude-sonnet-4-5-20250929 → global.anthropic.claude-sonnet-4-6
claude-haiku-4-5           → global.anthropic.claude-haiku-4-5-20251001-v1:0
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

## 安装

### 下载预编译二进制（推荐）

从 [Releases](https://github.com/KevinZhao/claudecode-bedrock-proxy/releases) 下载对应平台的二进制文件，无需安装 Go 环境：

```bash
# Linux x86_64
curl -Lo bedrock-effort-proxy https://github.com/KevinZhao/claudecode-bedrock-proxy/releases/latest/download/bedrock-effort-proxy-linux-amd64
chmod +x bedrock-effort-proxy

# Linux ARM64
curl -Lo bedrock-effort-proxy https://github.com/KevinZhao/claudecode-bedrock-proxy/releases/latest/download/bedrock-effort-proxy-linux-arm64

# macOS Apple Silicon
curl -Lo bedrock-effort-proxy https://github.com/KevinZhao/claudecode-bedrock-proxy/releases/latest/download/bedrock-effort-proxy-darwin-arm64

# macOS Intel
curl -Lo bedrock-effort-proxy https://github.com/KevinZhao/claudecode-bedrock-proxy/releases/latest/download/bedrock-effort-proxy-darwin-amd64
```

### 从源码构建

```bash
go build -o bedrock-effort-proxy .
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

### 适用场景

本 proxy 是一个通用的 Bedrock Claude API 反向代理，不仅限于 Claude Code，**任何使用 Anthropic Messages API 格式的应用**都可以通过它访问 Bedrock，并自动获得以下增强：

- **HTTP/2 上游连接** — Go 实现原生支持 HTTP/2 与 Bedrock 通信，相比 HTTP/1.1 显著降低延迟：多路复用消除队头阻塞，头部压缩减少每次请求开销，单连接复用避免重复 TLS 握手。对于 streaming 响应（SSE），HTTP/2 的帧级流控提供更稳定的数据传输。
- **自动 Prompt Caching 注入** — 对未携带 `cache_control` 的请求自动注入缓存断点，降低重复 input token 费用
- **Thinking/Effort 优化** — 自动为新模型启用 adaptive thinking + max effort
- **模型 ID 映射** — 透明转换 Anthropic 格式 model ID 为 Bedrock 跨区域推理 profile ID

适用示例：Cursor、Continue、Cline、自定义 AI 应用等任何支持配置 API base URL 的工具。

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

- Go 1.21+（仅从源码构建时需要，下载预编译二进制无需任何依赖）

### Python 版本（deprecated）

`proxy.py` 为早期 Python 实现，已废弃，不再维护。如需使用：

```bash
python3 proxy.py >> proxy.log 2>&1 &
```

依赖：Python 3.9+、`aiohttp`、`botocore`、`yarl`

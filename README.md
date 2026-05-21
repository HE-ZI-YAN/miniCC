# Mini Claude Code v3 - Production Agent

Mini Claude Code v3 是一个“生产级 AI Agent 系统”的最小工程骨架。它保留 v2 的 Planner/Executor/Reflector 思想，并升级为：

- Multi-Agent System
- Browser Agent
- GUI Agent
- MCP Tool System
- RAG
- Long-term Memory
- Task Queue
- Parallel Agent
- Event-driven Architecture
- Agent Trace

默认运行 v3：

```powershell
cc "帮我分析当前项目"
```

对比运行 v2：

```powershell
cc "帮我分析当前项目" --runtime v2
```

## 1. 真实工程目录

```text
miniCC/
  pyproject.toml
  README.md
  .env.example
  .gitignore
  mini_cc/
    agent.py             # v2 single-agent DAG
    production.py        # v3 stateful multi-agent DAG
    production_state.py  # Shared State / Agent message / Agent result
    events.py            # Event Bus / Agent Event / Tool Event / Streaming Event
    task_queue.py        # In-memory task queue
    memory.py            # Long-term memory JSON store
    rag.py               # Dependency-free codebase RAG baseline
    mcp.py               # MCP-like tool protocol/client/server
    browser_agent.py     # Browser Agent adapter
    gui_agent.py         # GUI Agent adapter
    observability.py     # Trace + cost tracker
    tools.py             # Local tools and patch system
    cli.py               # Typer + Rich CLI
    prompts.py           # v2 prompts
    state.py             # v2 state schema
```

## 2. 系统整体架构

```text
User Goal
  |
  v
Agent Router
  |
  +--> Planner Agent
  +--> Coding Agent  ----+
  +--> Test Agent    ----+--> Integrator
  +--> Reviewer Agent ---+
  +--> Debug Agent
  +--> Browser Agent
  +--> GUI Agent
  |
  v
Shared State + Event Bus + Long-term Memory + RAG + MCP Tools
```

核心实现：`mini_cc/production.py`。

v3 的 DAG：

```text
router -> parallel_agents -> integrator -> router
router -> END
integrator -> END
```

## 3. Multi-Agent 通信机制

通信对象在 `production_state.py`：

```python
class AgentMessage(BaseModel):
    sender: AgentRole
    recipient: AgentRole | Literal["router", "all"]
    content: str
    metadata: dict[str, Any]

class AgentAssignment(BaseModel):
    role: AgentRole
    task: str
    context: str
    parallel: bool

class AgentResult(BaseModel):
    role: AgentRole
    summary: str
    success: bool
    evidence: list[str]
    next_actions: list[AgentAssignment]
    tool_calls: list[dict[str, Any]]
```

生产系统里 Agent 间通常不直接互相调用，而是通过：

- Router 分派任务
- Shared State 共享上下文
- Event Bus 广播过程
- Queue 承载异步任务
- Integrator 汇总结果

## 4. Shared State

`SharedState` 是生产 Agent 的核心：

```python
class SharedState(BaseModel):
    run_id: str
    user_goal: str
    workspace: str
    todos: list[TaskItem]
    messages: list[AgentMessage]
    memory: MemoryState
    repository_context: list[str]
    assignments: list[AgentAssignment]
    results: list[AgentResult]
    artifacts: dict[str, Any]
    final_answer: str | None
    iteration: int
    max_iterations: int
    status: Literal["running", "done", "error"]
```

关键思想：生产级 Agent 不是一条 prompt，而是一个显式状态机。

## 5. Agent Router

Router 的职责：

- 根据用户目标、历史结果、长期记忆、MCP 工具列表选择下一批 Agent
- 决定哪些 Agent 并行运行
- 判断是否完成

当前默认路由：

```text
first round:
  Planner + Coding + Test

later rounds:
  Coding + Test + Reviewer
```

如果 LLM Router 输出合法 JSON，则按模型决策路由；否则使用 fallback route。

## 6. MCP 协议设计

`mini_cc/mcp.py` 实现了一个进程内 MCP-like 协议：

```text
tools/list
tools/call
```

请求：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "GitDiffTool",
    "arguments": {"path": "."}
  }
}
```

响应：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": "diff output",
  "error": null
}
```

生产环境可替换为真正的 MCP server process：

```text
Agent Runtime -> MCP Client -> MCP Server -> Tool Provider
```

## 7. Browser Agent

`browser_agent.py` 提供默认无依赖只读网页读取器：

- `fetch(url)`
- `search_url(query)`

生产替换建议：

- Playwright：强网页自动化、登录态、点击、表单
- Browser Use：自然语言浏览器操作
- Crawl4AI：文档抓取、站点爬取、Markdown 提取

生产 Browser Agent 架构：

```text
Browser Agent
  -> Search Provider
  -> Crawler
  -> HTML/Markdown Extractor
  -> RAG Index
  -> Citation Store
```

## 8. GUI Agent

`gui_agent.py` 是 pyautogui 可选适配层：

- `screenshot(path)`
- `click(x, y)`
- `type_text(text)`

生产 GUI Agent 需要三层：

```text
Screenshot
  -> Vision Model / Omni Parser
  -> UI Element Grounding
  -> pyautogui Action
  -> Observation Screenshot
```

关键不是“能点击”，而是每次点击后要观察屏幕并验证状态变化。

## 9. RAG 系统

`rag.py` 实现了一个无依赖 lexical RAG baseline：

- `index()` 扫描仓库
- `retrieve(query, k)` 返回相关文件片段

生产替换：

- FAISS：本地向量索引
- Chroma：轻量持久化向量库
- Qdrant：服务化向量数据库

生产级 RAG 结构：

```text
Code Chunker
  -> Embedding Model
  -> Vector DB
  -> Hybrid Retrieval
  -> Context Compressor
  -> Prompt Builder
```

## 10. Long-term Memory

`memory.py` 使用 `.mini_cc_memory.json` 保存：

- 用户偏好
- 历史任务
- 经验缓存
- 项目记忆

生产建议：

```text
Short-term Memory: current run state
Working Memory: scratchpad / active plan
Long-term Memory: project/user/task history
Semantic Memory: vectorized lessons and code facts
Episodic Memory: traces from previous runs
```

## 11. 并行执行架构

`production.py` 使用 `ThreadPoolExecutor`：

```text
parallel_agents node:
  Coding Agent
  Test Agent
  Reviewer Agent
  Browser Agent
  Debug Agent
```

生产环境建议拆成队列：

```text
Router -> Redis / RabbitMQ / NATS -> Worker Pool -> Result Store -> Integrator
```

并发不是为了热闹，而是为了让这些工作同时发生：

- 一个 Agent 分析代码
- 一个 Agent 跑测试
- 一个 Agent 查文档
- 一个 Agent review diff

## 12. Event System

`events.py` 定义：

- `AgentEvent`
- `EventBus`
- `agent.started`
- `agent.finished`
- `agent.routed`
- `tool.started`
- `tool.finished`
- `stream.token`
- `rag.indexed`
- `rag.retrieved`
- `mcp.tool_discovered`

事件流：

```text
Agent Runtime -> EventBus -> CLI Renderer
                         -> Trace Writer
                         -> Future Metrics Exporter
```

## 13. Streaming 架构

v3 中 streaming 不只来自 LLM token：

```text
LLM Token Event
Tool Event
Agent Event
Queue Event
RAG Event
Trace Event
```

CLI 统一消费 `v3_event`，按事件类型渲染。

## 14. 微服务拆分建议

生产环境建议拆分：

```text
api-gateway
agent-orchestrator
agent-worker-coding
agent-worker-browser
agent-worker-gui
mcp-tool-server
rag-indexer
memory-service
trace-service
web-ui
```

边界：

- Orchestrator 管 DAG 和 Shared State
- Workers 只执行 assignment
- MCP Server 管工具权限和审计
- RAG Indexer 异步维护索引
- Trace Service 存全链路事件

## 15. Docker 架构

建议：

```yaml
services:
  orchestrator:
    build: .
    env_file: .env
    depends_on: [redis, qdrant]

  worker-coding:
    build: .
    command: mini-cc-worker coding

  worker-browser:
    build: .
    command: mini-cc-worker browser

  redis:
    image: redis:7

  qdrant:
    image: qdrant/qdrant

  playwright:
    image: mcr.microsoft.com/playwright/python
```

## 16. 消息队列设计

队列主题：

```text
agent.assignments
agent.results
tool.requests
tool.results
events.stream
trace.events
```

消息要带：

- `run_id`
- `task_id`
- `agent_role`
- `attempt`
- `idempotency_key`
- `timeout`
- `budget`

## 17. 可观测性和日志系统

当前实现：

- `AgentTrace` 写 JSONL 到 `.omx/logs/mini_cc_trace_<run_id>.jsonl`
- `EventBus` 保留内存事件列表
- CLI 显示关键事件

生产建议：

- JSON structured logs
- OpenTelemetry traces
- Prometheus metrics
- per-agent latency
- per-tool failure rate
- token cost dashboard

## 18. Token 成本控制

`CostTracker` 当前记录 prompt/completion 字符数。

生产级策略：

- Prompt Cache：系统 prompt、工具 schema、repo summary 缓存
- Context Compression：长历史压缩成摘要
- RAG First：只取相关片段
- Model Router：简单任务用小模型，关键 review 用强模型
- Budget Guard：每个 run / task / agent 有 token 上限

## 19. Agent Trace

Trace 的价值：

- 回放一次 Agent 为什么这么做
- 复盘失败路径
- 做自动评测
- 训练更好的 Router/Prompt

v3 trace 是 JSONL：

```json
{"type": "agent.started", "source": "coding", "payload": {...}}
{"type": "tool.finished", "source": "RunTestTool", "payload": {...}}
```

## 20. Claude Code / Devin / Manus / OpenHands 共性架构

这些系统表面不同，但底层很相似：

1. **Agent Runtime**

   不是一次 LLM 调用，而是一个长期运行的状态机。

2. **Tool Sandbox**

   文件、shell、浏览器、GUI、网络都经过工具层，工具层负责权限、安全和审计。

3. **Planner + Executor 分离**

   Planner 管目标和任务树，Executor 负责下一步动作。

4. **Reflection / Verification Loop**

   测试、lint、review、浏览器观察会反过来改变计划。

5. **Long Context + RAG**

   大仓库不能全塞 prompt，必须检索、压缩、记忆。

6. **Event Trace**

   生产 Agent 必须能解释、回放、观测和调试。

7. **并行子任务**

   查文档、跑测试、代码分析、review 可以并行，最后由 Integrator 合并。

一句话：

> 生产级 AI Agent = Stateful DAG + Multi-Agent Router + Tool/MCP Sandbox + RAG/Memory + Event Trace + Verification Loop。

Mini Claude Code v3 就是这个架构的可运行最小模型。

# Mini Claude Code v1

一个最小可运行的 Coding Agent。它的目标不是复刻完整 Claude Code，而是把最核心的 Agent 循环做清楚：

```text
Thought
Action
Observation
Thought
Action
Observation
Final Answer
```

## 1. 项目目录结构

```text
miniCC/
  pyproject.toml
  README.md
  mini_cc/
    __init__.py
    __main__.py
    agent.py       # LangGraph ReAct loop
    cli.py         # Typer + Rich CLI
    prompts.py     # ReAct prompt template
    state.py       # LangGraph State design
    tools.py       # Tool schema + sandbox implementation
```

## 2. 完整 Agent 架构

```text
User CLI
  |
  v
cc "帮我分析当前项目"
  |
  v
MiniClaudeCodeAgent
  |
  +--> LangGraph StateGraph
        |
        +--> agent node
        |     - 发送 messages 给 OpenAI-compatible API
        |     - 流式打印 token
        |     - 解析 Thought / Action / Final Answer
        |
        +--> tool node
              - 校验工具名和 Pydantic args
              - 执行 sandbox tool
              - 把 Observation 写回 messages
```

Agent 每轮只做一件事：

1. 根据任务和历史 Observation 生成下一步。
2. 如果输出 `Action`，执行一个工具。
3. 把工具结果作为 `Observation` 送回模型。
4. 如果输出 `Final Answer`，循环结束。

## 3. LangGraph State 设计

见 `mini_cc/state.py`：

```python
class AgentState(TypedDict):
    task: str
    workspace: str
    messages: list[dict[str, str]]
    steps: list[AgentStep]
    pending_action: dict[str, Any] | None
    final_answer: str | None
    iteration: int
    max_iterations: int
    status: Literal["running", "tool", "done", "error"]
```

关键字段：

- `messages`：发给模型的完整上下文，包括 system、user、assistant 和 Observation。
- `steps`：结构化记录每一轮 ReAct，方便日志、回放和调试。
- `pending_action`：agent node 解析出的下一次工具调用。
- `status`：LangGraph 条件路由依据。

LangGraph 边：

```text
agent -> tool -> agent
agent -> END  when status == done/error
```

## 4. Tool Schema

所有工具在 `mini_cc/tools.py`，每个工具都有 Pydantic 输入模型。

```python
class ReadFileInput(BaseModel):
    path: str
    max_chars: int = 12000

class WriteFileInput(BaseModel):
    path: str
    content: str

class ListDirectoryInput(BaseModel):
    path: str = "."
    max_entries: int = 200

class ExecuteBashInput(BaseModel):
    command: str
    timeout_seconds: int = 30

class SearchCodeInput(BaseModel):
    keyword: str
    path: str = "."
    regex: bool = False
    max_matches: int = 50
```

已实现工具：

- `ReadFileTool`：读取文件内容
- `WriteFileTool`：写入文件
- `ListDirectoryTool`：查看目录结构
- `ExecuteBashTool`：执行 shell 命令
- `SearchCodeTool`：搜索代码关键字

## 5. Prompt 模板

见 `mini_cc/prompts.py`。

核心格式：

```text
Thought: your private reasoning summary for the next step
Action:
{"tool": "ToolName", "args": {"key": "value"}}

Thought: why the work is complete
Final Answer: concise final answer for the user
```

这里故意使用文本 ReAct，而不是 SDK tool calling。这样更容易看懂模型如何决定工具调用。

## 6. 核心代码

核心类是 `mini_cc/agent.py` 的 `MiniClaudeCodeAgent`。

重要方法：

- `_build_graph()`：创建 LangGraph 节点和边。
- `_agent_node()`：调用模型，流式接收 token，解析 Action 或 Final Answer。
- `_tool_node()`：执行工具，把 Observation 写回 state。
- `_stream_llm()`：OpenAI-compatible token streaming。

核心图结构：

```python
graph = StateGraph(AgentState)
graph.add_node("agent", self._agent_node)
graph.add_node("tool", self._tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    self._route_after_agent,
    {"tool": "tool", "done": END, "error": END},
)
graph.add_edge("tool", "agent")
```

## 7. Agent Loop 实现

一次循环的真实数据流：

```text
messages -> model
model streams text
parse Action JSON
pending_action = {"tool": "...", "args": {...}}
ToolRegistry.run(...)
Observation appended to messages
back to agent node
```

如果模型没有输出合法 Action，agent 会把格式错误作为 Observation 写回去，让模型自我修复。

如果超过 `max_iterations`，循环会以 error 状态结束，防止无限循环。

## 8. CLI 实现

安装依赖：

```bash
python -m pip install -e .
```

运行：

```bash
cc "帮我分析当前项目"
```

开发模式运行：

```bash
python -m mini_cc "帮我分析当前项目"
```

常用参数：

```bash
cc "给 README 增加安装说明" --workspace . --model gpt-4.1-mini --max-iterations 12
```

也可以指定 OpenAI 兼容接口：

```bash
cc "帮我分析当前项目" --base-url https://api.deepseek.com --model deepseek-v4-flash
```

## 9. DeepSeek 配置

可以用 DeepSeek，但不建议只把 DeepSeek key 塞进 `OPENAI_API_KEY` 后直接跑默认配置。

原因：

- SDK 可以继续用 OpenAI SDK。
- API 地址必须改成 DeepSeek 的 OpenAI 兼容地址：`https://api.deepseek.com`。
- 模型名也要换成 DeepSeek 模型，例如 `deepseek-v4-flash` 或 `deepseek-v4-pro`。

推荐方式：

```powershell
$env:DEEPSEEK_API_KEY="你的 deepseek key"
cc "帮我分析当前项目"
```

程序会自动使用：

```text
base_url = https://api.deepseek.com
model = deepseek-v4-flash
```

也可以显式指定：

```powershell
$env:OPENAI_API_KEY="你的 deepseek key"
$env:OPENAI_BASE_URL="https://api.deepseek.com"
cc "帮我分析当前项目" --model deepseek-v4-flash
```

OpenAI 用法仍然支持：

```powershell
$env:OPENAI_API_KEY="你的 openai key"
cc "帮我分析当前项目" --model gpt-4.1-mini
```

## 10. Streaming 实现

Streaming 分三层：

1. OpenAI-compatible API `stream=True` 持续返回 token。
2. `_stream_llm()` 每收到一个 token 就发出事件：`{"type": "token", "content": delta}`。
3. CLI 的 `render_event()` 用 Rich 实时打印：
   - `Thought #n`
   - token
   - `Action` 面板
   - `Observation` 面板
   - `Final Answer` 面板

## 11. 安全设计

MVP 做了三层限制：

1. 所有路径都会 resolve，并检查不能逃出 workspace。
2. `ExecuteBashTool` 固定在 workspace 下执行。
3. 拦截危险命令模式，例如：
   - `rm -rf`
   - `del /s`
   - `rmdir /s`
   - `format`
   - `mkfs`
   - `shutdown`
   - `reboot`
   - `reg delete`

这不是生产级 sandbox，但足够表达 Mini Claude Code 的安全边界思想：工具层必须比模型更可信。

## 12. MVP 开发顺序

推荐按这个顺序理解和继续扩展：

1. 先读 `tools.py`：Agent 的能力边界就是工具边界。
2. 再读 `prompts.py`：模型为什么会按 Thought / Action / Observation 输出。
3. 再读 `state.py`：循环中哪些信息需要持久化。
4. 再读 `agent.py`：LangGraph 如何把 agent node 和 tool node 连起来。
5. 最后读 `cli.py`：如何把 streaming 事件变成可视化终端体验。

下一步可以做：

- 把文本 Action 升级为 OpenAI tool calling。
- 增加 diff/patch 工具，而不是整文件写入。
- 给 ExecuteBashTool 增加 allowlist。
- 给每次运行保存 trace JSON。
- 支持 approval policy，对高风险写操作二次确认。

# Mini Claude Code v2

Mini Claude Code v2 是一个教学型 Coding Agent。v1 展示的是最小 ReAct 循环：

```text
Thought -> Action -> Observation -> Final Answer
```

v2 展示现代 Coding Agent 更核心的设计：

```text
User Goal
  -> Planner
  -> Executor
  -> Tool
  -> Reflector
  -> Planner / Executor
  -> Final Answer
```

也就是：先规划，再执行，再反思，再修正计划或重试，直到复杂任务完成。

## 1. 新架构设计

```text
                 ┌─────────────────────┐
                 │      User Goal       │
                 └──────────┬──────────┘
                            v
                 ┌─────────────────────┐
                 │   Planner Agent      │
                 │ - 拆解任务            │
                 │ - 维护 TODO           │
                 │ - 动态调整计划         │
                 └──────────┬──────────┘
                            v
                 ┌─────────────────────┐
                 │   Executor Agent     │
                 │ - 选择一个工具         │
                 │ - 执行当前 TODO        │
                 └──────────┬──────────┘
                            v
                 ┌─────────────────────┐
                 │      Tool Node       │
                 │ - 文件/搜索/命令/补丁   │
                 └──────────┬──────────┘
                            v
                 ┌─────────────────────┐
                 │  Reflection Agent    │
                 │ - 判断成败            │
                 │ - 诊断错误            │
                 │ - 决定 retry/plan     │
                 └──────┬────────┬─────┘
                        │        │
                        v        v
                    Executor   Planner
```

核心文件：

```text
mini_cc/
  agent.py       # LangGraph Stateful DAG
  state.py       # Task/Memory/Reflection schema
  prompts.py     # Planner/Executor/Reflector prompts
  tools.py       # Tools + patch system
  cli.py         # Typer + Rich streaming UI
```

## 2. LangGraph DAG

`mini_cc/agent.py` 中的 DAG：

```python
planner -> executor
executor -> tool | reflector
tool -> reflector
reflector -> executor | planner | END
```

节点职责：

- `planner`：理解用户目标，生成或调整 TODO List。
- `executor`：针对当前 TODO 选择一个工具调用，或声明当前 TODO 已完成。
- `tool`：执行真实外部动作，例如读文件、搜代码、跑测试、应用 patch。
- `reflector`：判断结果是否正确、是否完成、是否要重试或回到 planner。

条件边：

```text
planner:
  status == done  -> END
  otherwise       -> executor

executor:
  pending_action  -> tool
  no action       -> reflector

reflector:
  retry == true   -> executor
  all done        -> END
  otherwise       -> planner
```

## 3. State Schema

`mini_cc/state.py` 是 v2 的关键。现代 Agent 的能力来自“状态”，不是只来自 prompt。

```python
TaskStatus = Literal["TODO", "RUNNING", "BLOCKED", "DONE"]

class TaskItem(BaseModel):
    id: int
    title: str
    status: TaskStatus
    notes: str
    attempts: int

class MemoryState(BaseModel):
    conversation: list[dict[str, str]]
    task_memory: list[str]
    tool_history: list[dict[str, Any]]
    scratchpad: str

class AgentState(TypedDict):
    user_goal: str
    workspace: str
    todos: list[TaskItem]
    current_task_id: int | None
    memory: MemoryState
    steps: list[AgentStep]
    pending_action: dict[str, Any] | None
    last_observation: str | None
    last_executor_result: str | None
    last_reflection: ReflectionResult | None
    final_answer: str | None
    iteration: int
    max_iterations: int
    status: GraphStatus
```

这让 Agent 能回答几个关键问题：

- 当前目标是什么？
- 当前做到哪一个 TODO？
- 哪些工具调用过，结果是什么？
- 上一次失败原因是什么？
- 应该重试、换工具，还是重新规划？

## 4. Reflection Loop

Reflection 不是“总结一下”，而是控制流判断器。

Reflector 输出：

```json
{
  "task_status": "RUNNING",
  "success": false,
  "error_type": "test_failure",
  "diagnosis": "pytest failed because module import is wrong",
  "recovery_plan": "read the failing file and patch the import",
  "retry": true,
  "update_plan": false
}
```

如果 `retry=true`：

```text
Reflector -> Executor
```

如果 `update_plan=true`：

```text
Reflector -> Planner
```

如果所有 TODO 都是 `DONE`：

```text
Reflector -> END
```

这就是自动恢复的基础：错误不直接返回给用户，而是变成下一轮执行的上下文。

## 5. Planning 系统

Planner Prompt 在 `mini_cc/prompts.py`。

Planner 负责：

- 理解用户目标
- 拆解 TODO
- 选择当前任务
- 根据反思结果调整计划
- 判断整体是否完成

Planner 输出严格 JSON：

```json
{
  "thought": "Need to inspect the project before editing.",
  "todos": [
    {"id": 1, "title": "Inspect project structure", "status": "RUNNING", "notes": "", "attempts": 0},
    {"id": 2, "title": "Find duplicated code", "status": "TODO", "notes": "", "attempts": 0},
    {"id": 3, "title": "Apply small refactor patches", "status": "TODO", "notes": "", "attempts": 0},
    {"id": 4, "title": "Run tests and inspect diff", "status": "TODO", "notes": "", "attempts": 0}
  ],
  "current_task_id": 1,
  "scratchpad": "Refactor task; inspect first.",
  "done": false,
  "final_answer": null
}
```

## 6. Memory 设计

v2 实现了 5 类记忆：

- `Conversation Memory`：用户目标和对话上下文。
- `Task Memory`：每轮 reflection 产生的诊断记录。
- `Tool History`：工具名、参数、Observation。
- `Scratchpad`：Planner/Reflector 共享的短期工作区。
- `当前任务状态`：TODO/RUNNING/BLOCKED/DONE。

关键思想：不要指望模型“记得”。需要完成复杂任务，就要把关键状态显式持久化。

## 7. Patch 系统

v1 有 `WriteFileTool`，可以整文件覆盖。v2 新增 `ApplyPatchTool`，支持 unified diff：

```json
{
  "patch": "--- a/file.py\n+++ b/file.py\n@@ ...",
  "preview": true,
  "reverse": false
}
```

工作流：

1. `preview=true`：只做 `git apply --check`，展示 patch。
2. `preview=false`：真正应用 patch。
3. patch 路径会检查，不能逃出 workspace。

这更接近真实 Coding Agent：局部修改、可预览、可回滚、diff 可审查。

## 8. 错误恢复机制

错误来源：

- `TOOL_ERROR`
- shell 非 0 退出码
- 测试失败
- import 错误
- 文件不存在
- patch check/apply 失败

恢复路径：

```text
Tool Observation
  -> Reflector diagnosis
  -> retry=true
  -> Executor receives recovery_plan
  -> read/search/patch/test again
```

如果同一 TODO 多次失败，Planner 可以把它标成 `BLOCKED`，插入新的诊断任务，或者改变策略。

## 9. 新增工具

已有工具：

- `ReadFileTool`
- `WriteFileTool`
- `ListDirectoryTool`
- `ExecuteBashTool`
- `SearchCodeTool`

v2 新增：

- `GrepTool`：正则搜索文本。
- `ASTSearchTool`：搜索 Python AST 中的 class/function/import。
- `GitDiffTool`：查看当前 diff。
- `RunTestTool`：运行测试或验证命令。
- `ApplyPatchTool`：预览或应用 unified diff patch。

工具层仍然负责安全：

- 路径不能逃出 workspace。
- shell 拦截危险命令。
- patch 文件路径不能逃出 workspace。

## 10. Prompt 模板

### Planner Prompt

```text
You are the Planner Agent.
Turn the user's goal into an adaptive TODO list.
Do not call tools.
Return strict JSON with todos/current_task_id/done/final_answer.
```

### Executor Prompt

```text
You are the Executor Agent.
Execute exactly the current task using one tool call at a time.
Return strict JSON with action/action_input/task_complete/result.
Prefer ApplyPatchTool for edits.
```

### Reflection Prompt

```text
You are the Reflection Agent.
Judge whether the latest result moved the task forward.
Diagnose errors and decide retry or update_plan.
Return strict JSON with task_status/success/recovery_plan/retry.
```

完整文本见 `mini_cc/prompts.py`。

## 11. CLI

安装：

```bash
python -m pip install -e .
```

OpenAI：

```powershell
$env:OPENAI_API_KEY="你的 openai key"
cc "帮我重构当前项目" --model gpt-4.1-mini
```

DeepSeek：

```powershell
$env:DEEPSEEK_API_KEY="你的 deepseek key"
cc "帮我重构当前项目"
```

DeepSeek 默认使用：

```text
base_url = https://api.deepseek.com
model = deepseek-v4-flash
```

CLI 会流式显示：

- Planner token
- Plan 面板
- Executor token
- Tool Action 面板
- Observation 面板
- Reflection 面板
- Final Answer

## 12. Claude Code 的关键思想分析

Claude Code 这类现代 Coding Agent 的关键，不是“会调用工具”这么简单。

真正核心是四件事：

1. **显式状态**

   复杂任务不能只靠聊天上下文。Agent 必须维护 TODO、当前任务、工具历史、失败原因和 scratchpad。

2. **规划和执行分离**

   Planner 负责“做什么、按什么顺序做”，Executor 负责“下一步调用什么工具”。这能避免模型一边改代码一边忘记全局目标。

3. **反思控制流**

   Reflection 不是润色回答，而是决定 DAG 走向：retry、replan、done、blocked。

4. **工具层可信**

   模型可以犯错，所以文件路径、危险命令、patch 应用都必须在工具层校验。安全边界不能只写在 prompt 里。

一句话总结：

> 现代 Coding Agent = LLM 决策 + 显式状态机 + 可信工具层 + 反思恢复循环。

Mini Claude Code v2 就是这个设计的最小可运行版本。

from __future__ import annotations

import json
import os
import re
import warnings
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

warnings.filterwarnings("ignore", message=r".*allowed_objects.*")
try:
    from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

    warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)
except Exception:
    pass

from langgraph.graph import END, StateGraph
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from mini_cc.prompts import (
    EXECUTOR_PROMPT,
    PLANNER_PROMPT,
    REFLECTION_PROMPT,
    executor_user_prompt,
    planner_user_prompt,
    reflection_user_prompt,
)
from mini_cc.state import (
    AgentState,
    AgentStep,
    ExecutorDecision,
    MemoryState,
    PlannerDecision,
    ReflectionResult,
    TaskItem,
)
from mini_cc.tools import ToolError, ToolRegistry

EventSink = Callable[[dict[str, Any]], None]
T = TypeVar("T", bound=BaseModel)


class MiniClaudeCodeAgent:
    """Mini Claude Code v2: Planner -> Executor -> Reflector -> Planner."""

    def __init__(
        self,
        workspace: str,
        model: str,
        api_key: str,
        base_url: str | None = None,
        max_iterations: int = 20,
        event_sink: EventSink | None = None,
    ):
        self.workspace = workspace
        self.model = model
        self.max_iterations = max_iterations
        self.event_sink = event_sink or (lambda event: None)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.tools = ToolRegistry(workspace)
        self.graph = self._build_graph()

    def run(self, task: str) -> AgentState:
        initial: AgentState = {
            "user_goal": task,
            "workspace": self.workspace,
            "messages": [{"role": "user", "content": task}],
            "todos": [],
            "current_task_id": None,
            "memory": MemoryState(conversation=[{"role": "user", "content": task}]),
            "steps": [],
            "pending_action": None,
            "last_observation": None,
            "last_executor_result": None,
            "last_reflection": None,
            "final_answer": None,
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "status": "planning",
        }
        return self.graph.invoke(initial, config={"recursion_limit": self.max_iterations + 10})

    def stream(self, task: str) -> Iterator[dict[str, Any]]:
        events: list[dict[str, Any]] = []

        def capture(event: dict[str, Any]) -> None:
            events.append(event)
            self.event_sink(event)

        original_sink = self.event_sink
        self.event_sink = capture
        try:
            self.run(task)
            yield from events
        finally:
            self.event_sink = original_sink

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("planner", self._planner_node)
        graph.add_node("executor", self._executor_node)
        graph.add_node("tool", self._tool_node)
        graph.add_node("reflector", self._reflector_node)
        graph.set_entry_point("planner")
        graph.add_conditional_edges(
            "planner",
            self._route_after_planner,
            {"executor": "executor", "done": END, "error": END},
        )
        graph.add_conditional_edges(
            "executor",
            self._route_after_executor,
            {"tool": "tool", "reflector": "reflector", "error": END},
        )
        graph.add_edge("tool", "reflector")
        graph.add_conditional_edges(
            "reflector",
            self._route_after_reflector,
            {"planner": "planner", "executor": "executor", "done": END, "error": END},
        )
        return graph.compile()

    def _route_after_planner(self, state: AgentState) -> str:
        if state["status"] == "done":
            return "done"
        if state["status"] == "error":
            return "error"
        return "executor"

    def _route_after_executor(self, state: AgentState) -> str:
        if state["status"] == "error":
            return "error"
        if state["pending_action"]:
            return "tool"
        return "reflector"

    def _route_after_reflector(self, state: AgentState) -> str:
        if state["status"] == "done":
            return "done"
        if state["status"] == "error":
            return "error"
        reflection = state["last_reflection"]
        if reflection and reflection.retry:
            return "executor"
        return "planner"

    def _planner_node(self, state: AgentState) -> AgentState:
        if state["iteration"] >= state["max_iterations"]:
            return self._finish_with_error(state, "Reached max iterations before completion.")

        self.event_sink({"type": "planner_start", "iteration": state["iteration"] + 1})
        summary = _state_summary(state)
        content = self._stream_llm(
            [
                {"role": "system", "content": PLANNER_PROMPT},
                {"role": "user", "content": planner_user_prompt(summary)},
            ]
        )
        decision = _parse_model(content, PlannerDecision) or _fallback_plan(state, content)

        todos = decision.todos or state["todos"] or _seed_todos()
        current_task_id = decision.current_task_id or _next_todo_id(todos)
        todos = _mark_current_running(todos, current_task_id)
        memory = state["memory"].model_copy(update={"scratchpad": decision.scratchpad or state["memory"].scratchpad})
        steps = state["steps"] + [AgentStep(role="planner", content=decision.thought or content)]

        self.event_sink({"type": "plan", "todos": [todo.model_dump() for todo in todos], "current_task_id": current_task_id})

        if decision.done or _all_done(todos):
            final = decision.final_answer or _final_summary(state, todos)
            self.event_sink({"type": "final", "content": final})
            return {
                **state,
                "todos": todos,
                "current_task_id": None,
                "memory": memory,
                "steps": steps,
                "final_answer": final,
                "iteration": state["iteration"] + 1,
                "status": "done",
            }

        return {
            **state,
            "todos": todos,
            "current_task_id": current_task_id,
            "memory": memory,
            "steps": steps,
            "pending_action": None,
            "last_executor_result": None,
            "iteration": state["iteration"] + 1,
            "status": "executing",
        }

    def _executor_node(self, state: AgentState) -> AgentState:
        if state["iteration"] >= state["max_iterations"]:
            return self._finish_with_error(state, "Reached max iterations before completion.")

        self.event_sink({"type": "executor_start", "iteration": state["iteration"] + 1})
        summary = _state_summary(state)
        content = self._stream_llm(
            [
                {"role": "system", "content": EXECUTOR_PROMPT},
                {"role": "user", "content": executor_user_prompt(summary, self.tools.prompt_text())},
            ]
        )
        decision = _parse_model(content, ExecutorDecision)
        if not decision:
            decision = ExecutorDecision(
                thought="Executor response was not valid JSON; inspect the workspace as a recovery action.",
                action="ListDirectoryTool",
                action_input={"path": ".", "max_entries": 80},
            )

        steps = state["steps"] + [
            AgentStep(
                role="executor",
                content=decision.thought or content,
                action=decision.action,
                action_input=decision.action_input,
            )
        ]
        if decision.action:
            action = {"tool": decision.action, "args": decision.action_input}
            self.event_sink({"type": "tool_call", "tool": decision.action, "args": decision.action_input})
            return {
                **state,
                "steps": steps,
                "pending_action": action,
                "last_executor_result": None,
                "iteration": state["iteration"] + 1,
                "status": "executing",
            }

        result = decision.result or "Executor marked the current task complete without a tool call."
        self.event_sink({"type": "executor_result", "content": result})
        return {
            **state,
            "steps": steps,
            "pending_action": None,
            "last_executor_result": result,
            "iteration": state["iteration"] + 1,
            "status": "reflecting",
        }

    def _tool_node(self, state: AgentState) -> AgentState:
        action = state["pending_action"]
        if not action:
            return {**state, "status": "reflecting"}

        try:
            observation = self.tools.run(action["tool"], action.get("args", {}))
        except (ToolError, ValueError) as exc:
            observation = f"TOOL_ERROR: {exc}"

        self.event_sink({"type": "observation", "tool": action["tool"], "content": observation})
        memory = state["memory"].model_copy(
            update={
                "tool_history": state["memory"].tool_history
                + [{"tool": action["tool"], "args": action.get("args", {}), "observation": observation[:2000]}]
            }
        )
        steps = state["steps"] + [
            AgentStep(
                role="tool",
                content=f"{action['tool']} observation",
                action=action["tool"],
                action_input=action.get("args", {}),
                observation=observation,
            )
        ]
        return {
            **state,
            "memory": memory,
            "steps": steps,
            "pending_action": None,
            "last_observation": observation,
            "status": "reflecting",
        }

    def _reflector_node(self, state: AgentState) -> AgentState:
        if state["iteration"] >= state["max_iterations"]:
            return self._finish_with_error(state, "Reached max iterations before completion.")

        self.event_sink({"type": "reflection_start", "iteration": state["iteration"] + 1})
        summary = _state_summary(state)
        content = self._stream_llm(
            [
                {"role": "system", "content": REFLECTION_PROMPT},
                {"role": "user", "content": reflection_user_prompt(summary)},
            ]
        )
        reflection = _parse_model(content, ReflectionResult) or _fallback_reflection(state, content)
        todos = _update_current_task(state["todos"], state["current_task_id"], reflection)
        memory = _update_memory_after_reflection(state["memory"], reflection)
        steps = state["steps"] + [AgentStep(role="reflector", content=reflection.thought or content)]

        self.event_sink({"type": "reflection", "reflection": reflection.model_dump()})

        status = "planning"
        if reflection.retry:
            status = "executing"
        if _all_done(todos):
            final = _final_summary(state, todos)
            self.event_sink({"type": "final", "content": final})
            return {
                **state,
                "todos": todos,
                "memory": memory,
                "steps": steps,
                "last_reflection": reflection,
                "final_answer": final,
                "iteration": state["iteration"] + 1,
                "status": "done",
            }

        return {
            **state,
            "todos": todos,
            "memory": memory,
            "steps": steps,
            "last_reflection": reflection,
            "iteration": state["iteration"] + 1,
            "status": status,
        }

    def _stream_llm(self, messages: list[dict[str, str]]) -> str:
        chunks: list[str] = []
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=0,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                chunks.append(delta)
                self.event_sink({"type": "token", "content": delta})
        return "".join(chunks)

    def _finish_with_error(self, state: AgentState, message: str) -> AgentState:
        self.event_sink({"type": "final", "content": message})
        return {**state, "final_answer": message, "status": "error"}


def _parse_model(content: str, model: type[T]) -> T | None:
    raw = _extract_json_object(content)
    if not raw:
        return None
    try:
        return model.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError, TypeError):
        return None


def _extract_json_object(content: str) -> str | None:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.S)
    if fenced:
        return fenced.group(1)
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return content[start : end + 1]


def _state_summary(state: AgentState) -> str:
    todos = [todo.model_dump() for todo in state["todos"]]
    current = _current_task(state)
    payload = {
        "user_goal": state["user_goal"],
        "todos": todos,
        "current_task": current.model_dump() if current else None,
        "scratchpad": state["memory"].scratchpad,
        "recent_tool_history": state["memory"].tool_history[-5:],
        "last_observation": _trim(state["last_observation"]),
        "last_executor_result": state["last_executor_result"],
        "last_reflection": state["last_reflection"].model_dump() if state["last_reflection"] else None,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _fallback_plan(state: AgentState, content: str) -> PlannerDecision:
    todos = state["todos"] or _seed_todos()
    return PlannerDecision(
        thought=f"Planner fallback after invalid JSON: {_trim(content, 300)}",
        todos=todos,
        current_task_id=_next_todo_id(todos),
        scratchpad=state["memory"].scratchpad,
    )


def _fallback_reflection(state: AgentState, content: str) -> ReflectionResult:
    observation = state["last_observation"] or state["last_executor_result"] or content
    failed = "TOOL_ERROR" in observation or "exit_code=1" in observation or "FAILED" in observation
    return ReflectionResult(
        thought=f"Reflection fallback after invalid JSON: {_trim(content, 300)}",
        task_status="RUNNING" if failed else "DONE",
        success=not failed,
        error_type="tool_or_test_error" if failed else None,
        diagnosis=_trim(observation, 500),
        recovery_plan="Inspect the error, read the relevant file, and try a smaller corrective action." if failed else "",
        retry=failed,
        update_plan=not failed,
    )


def _seed_todos() -> list[TaskItem]:
    return [
        TaskItem(id=1, title="Inspect project structure", status="TODO"),
        TaskItem(id=2, title="Identify relevant files and implementation approach", status="TODO"),
        TaskItem(id=3, title="Make the smallest safe code changes", status="TODO"),
        TaskItem(id=4, title="Run verification and inspect diff", status="TODO"),
    ]


def _next_todo_id(todos: list[TaskItem]) -> int | None:
    for todo in todos:
        if todo.status in {"TODO", "RUNNING", "BLOCKED"}:
            return todo.id
    return None


def _mark_current_running(todos: list[TaskItem], current_task_id: int | None) -> list[TaskItem]:
    if current_task_id is None:
        return todos
    updated: list[TaskItem] = []
    for todo in todos:
        if todo.id == current_task_id and todo.status != "DONE":
            updated.append(todo.model_copy(update={"status": "RUNNING"}))
        else:
            updated.append(todo)
    return updated


def _update_current_task(
    todos: list[TaskItem],
    current_task_id: int | None,
    reflection: ReflectionResult,
) -> list[TaskItem]:
    if current_task_id is None:
        return todos
    updated: list[TaskItem] = []
    for todo in todos:
        if todo.id != current_task_id:
            updated.append(todo)
            continue
        notes = "\n".join(part for part in [todo.notes, reflection.diagnosis, reflection.recovery_plan] if part)
        updated.append(
            todo.model_copy(
                update={
                    "status": reflection.task_status,
                    "notes": _trim(notes, 1200),
                    "attempts": todo.attempts + 1,
                }
            )
        )
    return updated


def _update_memory_after_reflection(memory: MemoryState, reflection: ReflectionResult) -> MemoryState:
    task_memory = memory.task_memory
    if reflection.diagnosis:
        task_memory = task_memory + [_trim(reflection.diagnosis, 500)]
    scratchpad = memory.scratchpad
    if reflection.recovery_plan:
        scratchpad = _trim(f"{scratchpad}\nRecovery: {reflection.recovery_plan}", 2000)
    return memory.model_copy(update={"task_memory": task_memory[-20:], "scratchpad": scratchpad})


def _current_task(state: AgentState) -> TaskItem | None:
    for todo in state["todos"]:
        if todo.id == state["current_task_id"]:
            return todo
    return None


def _all_done(todos: list[TaskItem]) -> bool:
    return bool(todos) and all(todo.status == "DONE" for todo in todos)


def _final_summary(state: AgentState, todos: list[TaskItem]) -> str:
    done = [todo.title for todo in todos if todo.status == "DONE"]
    blocked = [todo.title for todo in todos if todo.status == "BLOCKED"]
    lines = ["Mini Claude Code v2 finished the task loop."]
    if done:
        lines.append("Done: " + "; ".join(done))
    if blocked:
        lines.append("Blocked: " + "; ".join(blocked))
    if state["last_observation"]:
        lines.append("Last observation: " + _trim(state["last_observation"], 500))
    return "\n".join(lines)


def _trim(value: str | None, limit: int = 1200) -> str | None:
    if value is None:
        return None
    return value if len(value) <= limit else value[:limit] + "\n...[truncated]"

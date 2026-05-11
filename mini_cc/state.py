from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

TaskStatus = Literal["TODO", "RUNNING", "BLOCKED", "DONE"]
GraphStatus = Literal["planning", "executing", "reflecting", "done", "error"]


class TaskItem(BaseModel):
    """One planner-managed unit of work."""

    id: int
    title: str
    status: TaskStatus = "TODO"
    notes: str = ""
    attempts: int = 0


class AgentStep(BaseModel):
    """One executor/tool/reflector step persisted in memory."""

    role: Literal["planner", "executor", "tool", "reflector"]
    content: str = ""
    action: str | None = None
    action_input: dict[str, Any] = Field(default_factory=dict)
    observation: str | None = None


class MemoryState(BaseModel):
    """Short-term memory that modern coding agents keep across the DAG."""

    conversation: list[dict[str, str]] = Field(default_factory=list)
    task_memory: list[str] = Field(default_factory=list)
    tool_history: list[dict[str, Any]] = Field(default_factory=list)
    scratchpad: str = ""


class PlannerDecision(BaseModel):
    thought: str = ""
    todos: list[TaskItem] = Field(default_factory=list)
    current_task_id: int | None = None
    scratchpad: str = ""
    done: bool = False
    final_answer: str | None = None


class ExecutorDecision(BaseModel):
    thought: str = ""
    action: str | None = None
    action_input: dict[str, Any] = Field(default_factory=dict)
    task_complete: bool = False
    result: str | None = None


class ReflectionResult(BaseModel):
    thought: str = ""
    task_status: TaskStatus = "RUNNING"
    success: bool = False
    error_type: str | None = None
    diagnosis: str = ""
    recovery_plan: str = ""
    retry: bool = False
    update_plan: bool = False


class AgentState(TypedDict):
    """LangGraph state carried by Mini Claude Code v2."""

    user_goal: str
    workspace: str
    messages: list[dict[str, str]]
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

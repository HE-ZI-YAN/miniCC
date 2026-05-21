from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

from mini_cc.state import MemoryState, TaskItem

AgentRole = Literal["planner", "coding", "test", "reviewer", "debug", "browser", "gui"]
TaskStatus = Literal["TODO", "RUNNING", "BLOCKED", "DONE"]


class AgentMessage(BaseModel):
    sender: AgentRole
    recipient: AgentRole | Literal["router", "all"]
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentAssignment(BaseModel):
    role: AgentRole
    task: str
    context: str = ""
    parallel: bool = False


class AgentResult(BaseModel):
    role: AgentRole
    summary: str
    success: bool = True
    evidence: list[str] = Field(default_factory=list)
    next_actions: list[AgentAssignment] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class SharedState(BaseModel):
    run_id: str
    user_goal: str
    workspace: str
    todos: list[TaskItem] = Field(default_factory=list)
    messages: list[AgentMessage] = Field(default_factory=list)
    memory: MemoryState = Field(default_factory=MemoryState)
    repository_context: list[str] = Field(default_factory=list)
    assignments: list[AgentAssignment] = Field(default_factory=list)
    results: list[AgentResult] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    final_answer: str | None = None
    iteration: int = 0
    max_iterations: int = 20
    status: Literal["running", "done", "error"] = "running"


class ProductionState(TypedDict):
    shared: SharedState
    route: str

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


class AgentStep(BaseModel):
    """One ReAct step persisted in LangGraph state."""

    thought: str = ""
    action: str | None = None
    action_input: dict[str, Any] = Field(default_factory=dict)
    observation: str | None = None


class AgentState(TypedDict):
    """LangGraph state carried between agent and tool nodes."""

    task: str
    workspace: str
    messages: list[dict[str, str]]
    steps: list[AgentStep]
    pending_action: dict[str, Any] | None
    final_answer: str | None
    iteration: int
    max_iterations: int
    status: Literal["running", "tool", "done", "error"]

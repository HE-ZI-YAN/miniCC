from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

EventType = Literal[
    "agent.started",
    "agent.finished",
    "agent.message",
    "agent.delegated",
    "agent.routed",
    "tool.started",
    "tool.finished",
    "tool.failed",
    "stream.token",
    "task.queued",
    "task.started",
    "task.finished",
    "memory.written",
    "rag.indexed",
    "rag.retrieved",
    "mcp.tool_discovered",
    "llm.error",
]


class AgentEvent(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    type: EventType
    source: str
    run_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


EventHandler = Callable[[AgentEvent], None]


class EventBus:
    """Small synchronous event bus used by the production runtime."""

    def __init__(self):
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self.events: list[AgentEvent] = []

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        self._handlers[event_type].append(handler)

    def publish(self, event: AgentEvent) -> None:
        self.events.append(event)
        for handler in self._handlers.get(event.type, []):
            handler(event)
        for handler in self._handlers.get("*", []):
            handler(event)

    def emit(self, event_type: EventType, source: str, run_id: str, **payload: Any) -> AgentEvent:
        event = AgentEvent(type=event_type, source=source, run_id=run_id, payload=payload)
        self.publish(event)
        return event

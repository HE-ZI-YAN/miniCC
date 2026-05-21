from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

from mini_cc.production_state import AgentAssignment

QueueStatus = Literal["queued", "running", "done", "failed"]


@dataclass
class QueueItem:
    assignment: AgentAssignment
    id: str = field(default_factory=lambda: uuid4().hex)
    status: QueueStatus = "queued"
    attempts: int = 0


class InMemoryTaskQueue:
    def __init__(self):
        self._items: deque[QueueItem] = deque()

    def push(self, assignment: AgentAssignment) -> QueueItem:
        item = QueueItem(assignment=assignment)
        self._items.append(item)
        return item

    def pop(self) -> QueueItem | None:
        if not self._items:
            return None
        item = self._items.popleft()
        item.status = "running"
        item.attempts += 1
        return item

    def drain(self) -> list[QueueItem]:
        items: list[QueueItem] = []
        while True:
            item = self.pop()
            if item is None:
                return items
            items.append(item)

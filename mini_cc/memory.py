from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class LongTermMemory:
    """JSON-backed project memory for preferences, tasks, and lessons."""

    def __init__(self, workspace: str | Path, filename: str = ".mini_cc_memory.json"):
        self.path = Path(workspace).resolve() / filename
        self.data = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"preferences": {}, "tasks": [], "lessons": [], "repository": {}}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"preferences": {}, "tasks": [], "lessons": [], "repository": {}}

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def remember_task(self, goal: str, summary: str, success: bool) -> None:
        self.data.setdefault("tasks", []).append(
            {
                "goal": goal,
                "summary": summary,
                "success": success,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.data["tasks"] = self.data["tasks"][-100:]
        self.save()

    def remember_lesson(self, lesson: str) -> None:
        self.data.setdefault("lessons", []).append(
            {"lesson": lesson, "created_at": datetime.now(timezone.utc).isoformat()}
        )
        self.data["lessons"] = self.data["lessons"][-100:]
        self.save()

    def summary(self) -> str:
        preferences = self.data.get("preferences", {})
        lessons = [item.get("lesson", "") for item in self.data.get("lessons", [])[-5:]]
        tasks = [item.get("summary", "") for item in self.data.get("tasks", [])[-5:]]
        return json.dumps(
            {"preferences": preferences, "recent_lessons": lessons, "recent_tasks": tasks},
            ensure_ascii=False,
            indent=2,
        )

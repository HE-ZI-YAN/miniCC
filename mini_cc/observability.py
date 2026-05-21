from __future__ import annotations

import json
from pathlib import Path

from mini_cc.events import AgentEvent


class AgentTrace:
    def __init__(self, workspace: str | Path, run_id: str):
        self.path = Path(workspace).resolve() / ".omx" / "logs" / f"mini_cc_trace_{run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: AgentEvent) -> None:
        with self.path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps(event.model_dump(), ensure_ascii=False) + "\n")


class CostTracker:
    """Approximate token budget tracker for prompt/cache decisions."""

    def __init__(self, max_prompt_chars: int = 24000):
        self.max_prompt_chars = max_prompt_chars
        self.prompt_chars = 0
        self.completion_chars = 0

    def add_prompt(self, text: str) -> None:
        self.prompt_chars += len(text)

    def add_completion(self, text: str) -> None:
        self.completion_chars += len(text)

    def should_compress(self) -> bool:
        return self.prompt_chars > self.max_prompt_chars

    def summary(self) -> str:
        return f"prompt_chars={self.prompt_chars}, completion_chars={self.completion_chars}"

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path


class CodebaseRAG:
    """Dependency-free lexical RAG baseline with a FAISS/Chroma-like interface."""

    def __init__(self, workspace: str | Path, max_file_bytes: int = 200_000):
        self.workspace = Path(workspace).resolve()
        self.max_file_bytes = max_file_bytes
        self.documents: list[tuple[str, str, Counter[str]]] = []

    def index(self) -> int:
        self.documents.clear()
        for path in self.workspace.rglob("*"):
            if not path.is_file() or _skip(path, self.max_file_bytes):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            rel = str(path.relative_to(self.workspace))
            self.documents.append((rel, text[:12000], Counter(_tokens(text))))
        return len(self.documents)

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        query_vector = Counter(_tokens(query))
        scored: list[tuple[float, str, str]] = []
        for rel, text, vector in self.documents:
            score = _cosine(query_vector, vector)
            if score > 0:
                scored.append((score, rel, text))
        scored.sort(reverse=True, key=lambda item: item[0])
        return [f"{rel}\n{_snippet(text, query)}" for _, rel, text in scored[:k]]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fff]+", text.lower())


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(left[token] * right.get(token, 0) for token in left)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _snippet(text: str, query: str, limit: int = 1200) -> str:
    query_tokens = _tokens(query)
    lowered = text.lower()
    index = 0
    for token in query_tokens:
        found = lowered.find(token)
        if found >= 0:
            index = found
            break
    start = max(0, index - 200)
    return text[start : start + limit]


def _skip(path: Path, max_file_bytes: int) -> bool:
    ignored = {".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache", ".mypy_cache"}
    if any(part in ignored for part in path.parts):
        return True
    try:
        return path.stat().st_size > max_file_bytes
    except OSError:
        return True

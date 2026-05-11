from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field


class ToolError(Exception):
    """Raised when a tool cannot safely complete."""


class ReadFileInput(BaseModel):
    path: str = Field(..., description="File path relative to the workspace.")
    max_chars: int = Field(12000, ge=1, le=50000)


class WriteFileInput(BaseModel):
    path: str = Field(..., description="File path relative to the workspace.")
    content: str = Field(..., description="Full file contents to write.")


class ListDirectoryInput(BaseModel):
    path: str = Field(".", description="Directory path relative to the workspace.")
    max_entries: int = Field(200, ge=1, le=1000)


class ExecuteBashInput(BaseModel):
    command: str = Field(..., description="Shell command to run inside the workspace.")
    timeout_seconds: int = Field(30, ge=1, le=120)


class SearchCodeInput(BaseModel):
    keyword: str = Field(..., min_length=1, description="Keyword or regex to search for.")
    path: str = Field(".", description="Directory path relative to the workspace.")
    regex: bool = Field(False, description="Treat keyword as a regular expression.")
    max_matches: int = Field(50, ge=1, le=500)


class ToolSpec(BaseModel):
    name: str
    description: str
    input_model: type[BaseModel]
    func: Callable[[BaseModel], str]

    def schema_for_prompt(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_model.model_json_schema(),
        }


class ToolRegistry:
    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve()
        self.tools: dict[str, ToolSpec] = {
            "ReadFileTool": ToolSpec(
                name="ReadFileTool",
                description="Read a text file from the sandbox workspace.",
                input_model=ReadFileInput,
                func=self._read_file,
            ),
            "WriteFileTool": ToolSpec(
                name="WriteFileTool",
                description="Write a text file inside the sandbox workspace.",
                input_model=WriteFileInput,
                func=self._write_file,
            ),
            "ListDirectoryTool": ToolSpec(
                name="ListDirectoryTool",
                description="List files and directories inside the sandbox workspace.",
                input_model=ListDirectoryInput,
                func=self._list_directory,
            ),
            "ExecuteBashTool": ToolSpec(
                name="ExecuteBashTool",
                description="Execute a shell command inside the sandbox workspace.",
                input_model=ExecuteBashInput,
                func=self._execute_bash,
            ),
            "SearchCodeTool": ToolSpec(
                name="SearchCodeTool",
                description="Search text files for a keyword or regex.",
                input_model=SearchCodeInput,
                func=self._search_code,
            ),
        }

    def prompt_text(self) -> str:
        lines = ["Available tools and JSON schemas:"]
        for tool in self.tools.values():
            lines.append(f"- {tool.schema_for_prompt()}")
        return "\n".join(lines)

    def run(self, name: str, args: dict[str, Any]) -> str:
        if name not in self.tools:
            raise ToolError(f"Unknown tool: {name}")
        spec = self.tools[name]
        parsed = spec.input_model.model_validate(args)
        return spec.func(parsed)

    def _safe_path(self, raw_path: str) -> Path:
        candidate = (self.workspace / raw_path).resolve()
        if candidate != self.workspace and self.workspace not in candidate.parents:
            raise ToolError(f"Path escapes sandbox: {raw_path}")
        return candidate

    def _read_file(self, data: BaseModel) -> str:
        args = data if isinstance(data, ReadFileInput) else ReadFileInput.model_validate(data)
        path = self._safe_path(args.path)
        if not path.is_file():
            raise ToolError(f"Not a file: {args.path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        suffix = "" if len(text) <= args.max_chars else "\n...[truncated]"
        return text[: args.max_chars] + suffix

    def _write_file(self, data: BaseModel) -> str:
        args = data if isinstance(data, WriteFileInput) else WriteFileInput.model_validate(data)
        path = self._safe_path(args.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args.content, encoding="utf-8")
        return f"Wrote {len(args.content)} characters to {path.relative_to(self.workspace)}"

    def _list_directory(self, data: BaseModel) -> str:
        args = data if isinstance(data, ListDirectoryInput) else ListDirectoryInput.model_validate(data)
        path = self._safe_path(args.path)
        if not path.is_dir():
            raise ToolError(f"Not a directory: {args.path}")

        rows: list[str] = []
        for index, item in enumerate(sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))):
            if index >= args.max_entries:
                rows.append("...[truncated]")
                break
            kind = "dir " if item.is_dir() else "file"
            rows.append(f"{kind} {item.relative_to(self.workspace)}")
        return "\n".join(rows) or "(empty directory)"

    def _execute_bash(self, data: BaseModel) -> str:
        args = data if isinstance(data, ExecuteBashInput) else ExecuteBashInput.model_validate(data)
        self._guard_command(args.command)
        result = subprocess.run(
            args.command,
            cwd=self.workspace,
            shell=True,
            text=True,
            capture_output=True,
            timeout=args.timeout_seconds,
        )
        output = result.stdout
        if result.stderr:
            output += ("\nSTDERR:\n" if output else "STDERR:\n") + result.stderr
        return f"exit_code={result.returncode}\n{output}".strip()

    def _search_code(self, data: BaseModel) -> str:
        args = data if isinstance(data, SearchCodeInput) else SearchCodeInput.model_validate(data)
        root = self._safe_path(args.path)
        if not root.exists():
            raise ToolError(f"Path does not exist: {args.path}")

        pattern = re.compile(args.keyword) if args.regex else None
        matches: list[str] = []
        files = [root] if root.is_file() else (p for p in root.rglob("*") if p.is_file())
        for file_path in files:
            if len(matches) >= args.max_matches:
                break
            if _skip_file(file_path):
                continue
            try:
                lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            for line_no, line in enumerate(lines, start=1):
                ok = bool(pattern.search(line)) if pattern else args.keyword in line
                if ok:
                    rel = file_path.relative_to(self.workspace)
                    matches.append(f"{rel}:{line_no}: {line[:240]}")
                    if len(matches) >= args.max_matches:
                        break
        return "\n".join(matches) or "(no matches)"

    def _guard_command(self, command: str) -> None:
        normalized = command.lower().strip()
        dangerous_patterns = [
            r"\brm\s+-[^\n;|&]*r[^\n;|&]*f\b",
            r"\brm\s+-[^\n;|&]*f[^\n;|&]*r\b",
            r"\bdel\s+/[sq]\b",
            r"\brmdir\s+/s\b",
            r"\bformat\b",
            r"\bmkfs\b",
            r"\bshutdown\b",
            r"\breboot\b",
            r"\breg\s+delete\b",
            r":\s*\(\s*\)\s*\{",
        ]
        if any(re.search(pattern, normalized) for pattern in dangerous_patterns):
            raise ToolError(f"Blocked dangerous command: {command}")

        destructive_roots = ["/", "c:\\", "d:\\", str(self.workspace.anchor).lower()]
        for root in destructive_roots:
            if normalized in {f"rm -rf {root}", f"remove-item -recurse -force {root}"}:
                raise ToolError(f"Blocked destructive root command: {command}")


def _skip_file(path: Path) -> bool:
    ignored_parts = {".git", ".venv", "venv", "__pycache__", "node_modules", ".mypy_cache", ".pytest_cache"}
    if any(part in ignored_parts for part in path.parts):
        return True
    try:
        return path.stat().st_size > 1_000_000
    except OSError:
        return True

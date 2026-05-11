from __future__ import annotations

import ast
import os
import re
import subprocess
import tempfile
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


class GrepInput(BaseModel):
    pattern: str = Field(..., min_length=1, description="Regex pattern to search for.")
    path: str = Field(".", description="Directory or file path relative to the workspace.")
    max_matches: int = Field(100, ge=1, le=1000)


class ASTSearchInput(BaseModel):
    pattern: str = Field(..., min_length=1, description="Name substring or regex to match.")
    path: str = Field(".", description="Python file or directory relative to the workspace.")
    node_types: list[str] = Field(
        default_factory=lambda: ["ClassDef", "FunctionDef", "AsyncFunctionDef", "Import", "ImportFrom"],
        description="AST node class names to include.",
    )
    regex: bool = Field(False, description="Treat pattern as regex.")
    max_matches: int = Field(100, ge=1, le=1000)


class GitDiffInput(BaseModel):
    path: str = Field(".", description="Path relative to the workspace.")
    staged: bool = Field(False, description="Show staged diff.")
    max_chars: int = Field(20000, ge=1, le=100000)


class RunTestInput(BaseModel):
    command: str = Field("python -m pytest", description="Test command to run in the workspace.")
    timeout_seconds: int = Field(120, ge=1, le=600)


class ApplyPatchInput(BaseModel):
    patch: str = Field(..., description="Unified diff patch.")
    preview: bool = Field(True, description="When true, validate and preview without applying.")
    reverse: bool = Field(False, description="Apply the patch in reverse.")


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
            "GrepTool": ToolSpec(
                name="GrepTool",
                description="Regex grep across text files with file:line output.",
                input_model=GrepInput,
                func=self._grep,
            ),
            "ASTSearchTool": ToolSpec(
                name="ASTSearchTool",
                description="Search Python AST symbols such as classes, functions, and imports.",
                input_model=ASTSearchInput,
                func=self._ast_search,
            ),
            "GitDiffTool": ToolSpec(
                name="GitDiffTool",
                description="Show git diff for the workspace or a path.",
                input_model=GitDiffInput,
                func=self._git_diff,
            ),
            "RunTestTool": ToolSpec(
                name="RunTestTool",
                description="Run a test or verification command inside the workspace.",
                input_model=RunTestInput,
                func=self._run_test,
            ),
            "ApplyPatchTool": ToolSpec(
                name="ApplyPatchTool",
                description="Validate, preview, or apply a unified diff patch.",
                input_model=ApplyPatchInput,
                func=self._apply_patch,
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
            encoding="utf-8",
            errors="replace",
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

    def _grep(self, data: BaseModel) -> str:
        args = data if isinstance(data, GrepInput) else GrepInput.model_validate(data)
        root = self._safe_path(args.path)
        if not root.exists():
            raise ToolError(f"Path does not exist: {args.path}")
        try:
            pattern = re.compile(args.pattern)
        except re.error as exc:
            raise ToolError(f"Invalid regex: {exc}") from exc

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
                if pattern.search(line):
                    matches.append(f"{file_path.relative_to(self.workspace)}:{line_no}: {line[:240]}")
                    if len(matches) >= args.max_matches:
                        break
        return "\n".join(matches) or "(no matches)"

    def _ast_search(self, data: BaseModel) -> str:
        args = data if isinstance(data, ASTSearchInput) else ASTSearchInput.model_validate(data)
        root = self._safe_path(args.path)
        if not root.exists():
            raise ToolError(f"Path does not exist: {args.path}")
        matcher = re.compile(args.pattern).search if args.regex else lambda value: args.pattern in value
        node_types = set(args.node_types)
        files = [root] if root.is_file() else (p for p in root.rglob("*.py") if p.is_file())
        matches: list[str] = []

        for file_path in files:
            if len(matches) >= args.max_matches or _skip_file(file_path):
                continue
            try:
                tree = ast.parse(file_path.read_text(encoding="utf-8", errors="ignore"))
            except (SyntaxError, OSError):
                continue
            for node in ast.walk(tree):
                if len(matches) >= args.max_matches:
                    break
                node_type = type(node).__name__
                if node_type not in node_types:
                    continue
                label = _ast_label(node)
                if label and matcher(label):
                    line_no = getattr(node, "lineno", 1)
                    matches.append(f"{file_path.relative_to(self.workspace)}:{line_no}: {node_type} {label}")
        return "\n".join(matches) or "(no matches)"

    def _git_diff(self, data: BaseModel) -> str:
        args = data if isinstance(data, GitDiffInput) else GitDiffInput.model_validate(data)
        path = self._safe_path(args.path)
        command = ["git", "diff"]
        if args.staged:
            command.append("--staged")
        command.extend(["--", str(path.relative_to(self.workspace))])
        result = subprocess.run(
            command,
            cwd=self.workspace,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=30,
        )
        output = result.stdout or result.stderr or "(no diff)"
        suffix = "" if len(output) <= args.max_chars else "\n...[truncated]"
        return f"exit_code={result.returncode}\n{output[: args.max_chars]}{suffix}".strip()

    def _run_test(self, data: BaseModel) -> str:
        args = data if isinstance(data, RunTestInput) else RunTestInput.model_validate(data)
        self._guard_command(args.command)
        result = subprocess.run(
            args.command,
            cwd=self.workspace,
            shell=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=args.timeout_seconds,
        )
        output = result.stdout
        if result.stderr:
            output += ("\nSTDERR:\n" if output else "STDERR:\n") + result.stderr
        return f"exit_code={result.returncode}\n{output}".strip()

    def _apply_patch(self, data: BaseModel) -> str:
        args = data if isinstance(data, ApplyPatchInput) else ApplyPatchInput.model_validate(data)
        _validate_patch_paths(args.patch, self.workspace)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".patch") as patch_file:
            patch_file.write(args.patch)
            patch_path = patch_file.name
        try:
            check_command = ["git", "apply", "--check", "--whitespace=nowarn"]
            if args.reverse:
                check_command.append("--reverse")
            check_command.append(patch_path)
            check = subprocess.run(
                check_command,
                cwd=self.workspace,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=30,
            )
            if check.returncode != 0:
                return f"PATCH_CHECK_FAILED\n{check.stderr or check.stdout}".strip()
            if args.preview:
                return f"PATCH_PREVIEW_OK\n{args.patch}"

            apply_command = ["git", "apply", "--whitespace=nowarn"]
            if args.reverse:
                apply_command.append("--reverse")
            apply_command.append(patch_path)
            applied = subprocess.run(
                apply_command,
                cwd=self.workspace,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=30,
            )
            if applied.returncode != 0:
                return f"PATCH_APPLY_FAILED\n{applied.stderr or applied.stdout}".strip()
            return "PATCH_APPLIED"
        finally:
            try:
                os.unlink(patch_path)
            except OSError:
                pass

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


def _ast_label(node: ast.AST) -> str:
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name
    if isinstance(node, ast.Import):
        return ", ".join(alias.name for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        names = ", ".join(alias.name for alias in node.names)
        return f"{module}: {names}"
    return ""


def _validate_patch_paths(patch: str, workspace: Path) -> None:
    for line in patch.splitlines():
        if not (line.startswith("--- ") or line.startswith("+++ ")):
            continue
        raw = line[4:].strip().split("\t", 1)[0]
        if raw == "/dev/null":
            continue
        if raw.startswith("a/") or raw.startswith("b/"):
            raw = raw[2:]
        candidate = (workspace / raw).resolve()
        if candidate != workspace and workspace not in candidate.parents:
            raise ToolError(f"Patch path escapes sandbox: {raw}")

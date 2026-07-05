from __future__ import annotations

import os
from typing import Any

import typer
from openai import APIConnectionError, APIStatusError, RateLimitError
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
OPENAI_DEFAULT_MODEL = "gpt-4.1-mini"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash"


def entrypoint() -> None:
    typer.run(main)


def main(
    task: str = typer.Argument(..., help='Natural language task, e.g. cc "analyze this project"'),
    workspace: str = typer.Option(".", "--workspace", "-w", help="Sandbox working directory."),
    model: str | None = typer.Option(None, "--model", "-m", help="Model name."),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        help="OpenAI-compatible API base URL. Defaults to DeepSeek when DEEPSEEK_API_KEY is set.",
    ),
    max_iterations: int = typer.Option(30, "--max-iterations", help="Maximum DAG node iterations."),
    runtime: str = typer.Option("v3", "--runtime", help="Agent runtime: v2 or v3."),
    parallelism: int = typer.Option(3, "--parallelism", help="Parallel agents for v3 runtime."),
) -> None:
    """Run the coding agent."""
    workspace_path = os.path.abspath(workspace)
    load_env_file(os.getcwd())
    load_env_file(workspace_path)
    resolved_api_key, resolved_base_url, resolved_model = resolve_llm_config(model, base_url)

    console.print(Panel.fit(f"[bold]Task[/bold]\n{task}", border_style="cyan"))
    if runtime.lower() == "v2":
        from mini_cc.agent import MiniClaudeCodeAgent

        agent = MiniClaudeCodeAgent(
            workspace=workspace_path,
            model=resolved_model,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            max_iterations=max_iterations,
            event_sink=render_event,
        )
        try:
            agent.run(task)
        except (APIStatusError, APIConnectionError, RateLimitError) as exc:
            render_provider_error(exc)
        return

    if runtime.lower() != "v3":
        raise typer.BadParameter("--runtime must be v2 or v3")

    from mini_cc.production import ProductionAgentRuntime

    agent = ProductionAgentRuntime(
        workspace=workspace_path,
        model=resolved_model,
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        max_iterations=max_iterations,
        parallelism=parallelism,
        event_sink=render_event,
    )
    try:
        agent.run(task)
    except (APIStatusError, APIConnectionError, RateLimitError) as exc:
        render_provider_error(exc)


def render_event(event: dict[str, Any]) -> None:
    event_type = event["type"]
    if event_type == "planner_start":
        console.print(f"\n[bold blue]Planner #{event['iteration']}[/bold blue]")
    elif event_type == "executor_start":
        console.print(f"\n[bold magenta]Executor #{event['iteration']}[/bold magenta]")
    elif event_type == "reflection_start":
        console.print(f"\n[bold green]Reflector #{event['iteration']}[/bold green]")
    elif event_type == "token":
        console.print(event["content"], end="", soft_wrap=True)
    elif event_type == "plan":
        lines = []
        for todo in event["todos"]:
            marker = "->" if todo["id"] == event["current_task_id"] else "  "
            lines.append(f"{marker} [{todo['status']}] {todo['id']}. {todo['title']}")
        console.print()
        console.print(Panel("\n".join(lines), title="Plan", border_style="blue"))
    elif event_type == "tool_call":
        console.print()
        payload = {"tool": event["tool"], "args": event["args"]}
        syntax = Syntax(str(payload), "python", theme="ansi_dark", word_wrap=True)
        console.print(Panel(syntax, title="Action", border_style="yellow"))
    elif event_type == "observation":
        console.print(Panel(event["content"], title=f"Observation: {event['tool']}", border_style="green"))
    elif event_type == "executor_result":
        console.print(Panel(event["content"], title="Executor Result", border_style="magenta"))
    elif event_type == "reflection":
        reflection = event["reflection"]
        content = "\n".join(
            [
                f"status: {reflection['task_status']}",
                f"success: {reflection['success']}",
                f"retry: {reflection['retry']}",
                f"update_plan: {reflection['update_plan']}",
                f"diagnosis: {reflection['diagnosis']}",
                f"recovery: {reflection['recovery_plan']}",
            ]
        )
        console.print()
        console.print(Panel(content, title="Reflection", border_style="green"))
    elif event_type == "final":
        console.print()
        console.print(Panel(event["content"], title="Final Answer", border_style="cyan"))
    elif event_type == "provider_error":
        console.print()
        console.print(Panel(event["content"], title="LLM Provider Error", border_style="red"))
    elif event_type == "v3_event":
        event_payload = event["event"]
        name = event_payload["type"]
        source = event_payload["source"]
        payload = event_payload.get("payload", {})
        if name == "stream.token":
            console.print(payload.get("token", ""), end="", soft_wrap=True)
        elif name == "llm.error":
            console.print()
            console.print(Panel(payload.get("message", "LLM provider error"), title="LLM Error", border_style="red"))
        elif name in {"agent.started", "agent.routed", "agent.finished", "tool.started", "tool.finished", "tool.failed"}:
            console.print()
            console.print(Panel(str(payload), title=f"{name} :: {source}", border_style="cyan"))


def resolve_llm_config(model: str | None, base_url: str | None) -> tuple[str, str | None, str]:
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    env_base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL")

    if deepseek_key and not openai_key:
        return (
            deepseek_key,
            base_url or env_base_url or DEEPSEEK_BASE_URL,
            model or DEEPSEEK_DEFAULT_MODEL,
        )
    if openai_key:
        return openai_key, base_url or env_base_url, model or OPENAI_DEFAULT_MODEL
    if deepseek_key:
        return (
            deepseek_key,
            base_url or env_base_url or DEEPSEEK_BASE_URL,
            model or DEEPSEEK_DEFAULT_MODEL,
        )

    raise typer.BadParameter("Set OPENAI_API_KEY or DEEPSEEK_API_KEY.")


def load_env_file(directory: str) -> None:
    env_path = os.path.join(directory, ".env")
    if not os.path.isfile(env_path):
        return

    with open(env_path, encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def render_provider_error(exc: Exception) -> None:
    message = provider_error_message(exc)
    console.print()
    console.print(Panel(message, title="LLM Provider Error", border_style="red"))


def provider_error_message(exc: Exception) -> str:
    if isinstance(exc, APIStatusError):
        body = getattr(exc, "body", None)
        provider_message = ""
        if isinstance(body, dict):
            error = body.get("error") or {}
            if isinstance(error, dict):
                provider_message = str(error.get("message") or "")
        if exc.status_code == 402 or "insufficient balance" in provider_message.lower():
            return (
                "API 返回 402：Insufficient Balance。\n\n"
                "原因：当前 API key 对应的账户余额不足，不是 Mini Claude Code 的工具执行错误。\n\n"
                "处理方式：\n"
                "1. 去 DeepSeek/OpenAI 控制台充值或更换有余额的 key。\n"
                "2. 检查 .env 中是否配置了正确的 DEEPSEEK_API_KEY 或 OPENAI_API_KEY。\n"
                "3. 如果你想临时避免调用模型，可以用 v3 的本地 fallback 结果作为参考，但真正智能路由仍需要可用余额。"
            )
        return f"API 状态错误 {exc.status_code}: {provider_message or exc}"
    if isinstance(exc, RateLimitError):
        return "API 触发限流，请稍后重试或降低并发/切换模型。"
    if isinstance(exc, APIConnectionError):
        return "API 网络连接失败，请检查网络、base_url 和代理配置。"
    return f"模型供应商调用失败：{exc}"

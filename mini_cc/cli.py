from __future__ import annotations

import os
from typing import Any

import typer
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
    max_iterations: int = typer.Option(12, "--max-iterations", help="Maximum ReAct turns."),
) -> None:
    """Run the coding agent."""
    resolved_api_key, resolved_base_url, resolved_model = resolve_llm_config(model, base_url)

    from mini_cc.agent import MiniClaudeCodeAgent

    console.print(Panel.fit(f"[bold]Task[/bold]\n{task}", border_style="cyan"))
    agent = MiniClaudeCodeAgent(
        workspace=os.path.abspath(workspace),
        model=resolved_model,
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        max_iterations=max_iterations,
        event_sink=render_event,
    )
    agent.run(task)


def render_event(event: dict[str, Any]) -> None:
    event_type = event["type"]
    if event_type == "thought_start":
        console.print(f"\n[bold magenta]Thought #{event['iteration']}[/bold magenta]")
    elif event_type == "token":
        console.print(event["content"], end="", soft_wrap=True)
    elif event_type == "tool_call":
        console.print()
        payload = {"tool": event["tool"], "args": event["args"]}
        syntax = Syntax(str(payload), "python", theme="ansi_dark", word_wrap=True)
        console.print(Panel(syntax, title="Action", border_style="yellow"))
    elif event_type == "observation":
        console.print(Panel(event["content"], title=f"Observation: {event['tool']}", border_style="green"))
    elif event_type == "final":
        console.print()
        console.print(Panel(event["content"], title="Final Answer", border_style="cyan"))


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

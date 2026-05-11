from __future__ import annotations

import json
import os
import re
import warnings
from collections.abc import Callable, Iterator
from typing import Any

warnings.filterwarnings(
    "ignore",
    message=r".*allowed_objects.*",
)
try:
    from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

    warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)
except Exception:
    pass

from langgraph.graph import END, StateGraph
from openai import OpenAI

from mini_cc.prompts import SYSTEM_PROMPT, user_prompt
from mini_cc.state import AgentState, AgentStep
from mini_cc.tools import ToolError, ToolRegistry

EventSink = Callable[[dict[str, Any]], None]


class MiniClaudeCodeAgent:
    def __init__(
        self,
        workspace: str,
        model: str,
        api_key: str,
        base_url: str | None = None,
        max_iterations: int = 12,
        event_sink: EventSink | None = None,
    ):
        self.workspace = workspace
        self.model = model
        self.max_iterations = max_iterations
        self.event_sink = event_sink or (lambda event: None)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.tools = ToolRegistry(workspace)
        self.graph = self._build_graph()

    def run(self, task: str) -> AgentState:
        initial: AgentState = {
            "task": task,
            "workspace": self.workspace,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + self.tools.prompt_text()},
                {"role": "user", "content": user_prompt(task)},
            ],
            "steps": [],
            "pending_action": None,
            "final_answer": None,
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "status": "running",
        }
        return self.graph.invoke(initial)

    def stream(self, task: str) -> Iterator[dict[str, Any]]:
        events: list[dict[str, Any]] = []

        def capture(event: dict[str, Any]) -> None:
            events.append(event)
            self.event_sink(event)

        original_sink = self.event_sink
        self.event_sink = capture
        try:
            self.run(task)
            yield from events
        finally:
            self.event_sink = original_sink

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tool", self._tool_node)
        graph.set_entry_point("agent")
        graph.add_conditional_edges(
            "agent",
            self._route_after_agent,
            {"tool": "tool", "done": END, "error": END},
        )
        graph.add_edge("tool", "agent")
        return graph.compile()

    def _route_after_agent(self, state: AgentState) -> str:
        if state["status"] == "done":
            return "done"
        if state["status"] == "error":
            return "error"
        return "tool"

    def _agent_node(self, state: AgentState) -> AgentState:
        if state["iteration"] >= state["max_iterations"]:
            final = "Reached max iterations before completion."
            self.event_sink({"type": "final", "content": final})
            return {**state, "final_answer": final, "status": "error"}

        self.event_sink({"type": "thought_start", "iteration": state["iteration"] + 1})
        content = self._stream_llm(state["messages"])
        thought = _extract_thought(content)
        final_answer = _extract_final_answer(content)
        action = _extract_action(content)

        if final_answer:
            self.event_sink({"type": "final", "content": final_answer})
            return {
                **state,
                "messages": state["messages"] + [{"role": "assistant", "content": content}],
                "steps": state["steps"] + [AgentStep(thought=thought)],
                "pending_action": None,
                "final_answer": final_answer,
                "iteration": state["iteration"] + 1,
                "status": "done",
            }

        if not action:
            repair = (
                "Your previous response did not contain a valid Action JSON object. "
                "Continue using the required ReAct format."
            )
            return {
                **state,
                "messages": state["messages"]
                + [{"role": "assistant", "content": content}, {"role": "user", "content": repair}],
                "steps": state["steps"] + [AgentStep(thought=thought, observation=repair)],
                "pending_action": None,
                "iteration": state["iteration"] + 1,
                "status": "running",
            }

        self.event_sink({"type": "tool_call", "tool": action["tool"], "args": action["args"]})
        return {
            **state,
            "messages": state["messages"] + [{"role": "assistant", "content": content}],
            "steps": state["steps"]
            + [AgentStep(thought=thought, action=action["tool"], action_input=action["args"])],
            "pending_action": action,
            "iteration": state["iteration"] + 1,
            "status": "tool",
        }

    def _tool_node(self, state: AgentState) -> AgentState:
        action = state["pending_action"]
        if not action:
            return {**state, "status": "running"}

        try:
            observation = self.tools.run(action["tool"], action.get("args", {}))
        except (ToolError, ValueError) as exc:
            observation = f"TOOL_ERROR: {exc}"

        self.event_sink({"type": "observation", "tool": action["tool"], "content": observation})
        steps = list(state["steps"])
        if steps:
            steps[-1] = steps[-1].model_copy(update={"observation": observation})

        return {
            **state,
            "messages": state["messages"] + [{"role": "user", "content": f"Observation:\n{observation}"}],
            "steps": steps,
            "pending_action": None,
            "status": "running",
        }

    def _stream_llm(self, messages: list[dict[str, str]]) -> str:
        chunks: list[str] = []
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=0,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                chunks.append(delta)
                self.event_sink({"type": "token", "content": delta})
        return "".join(chunks)


def _extract_thought(content: str) -> str:
    match = re.search(r"Thought:\s*(.*?)(?:\nAction:|\nFinal Answer:|$)", content, re.S)
    return match.group(1).strip() if match else ""


def _extract_final_answer(content: str) -> str | None:
    match = re.search(r"Final Answer:\s*(.*)$", content, re.S)
    return match.group(1).strip() if match else None


def _extract_action(content: str) -> dict[str, Any] | None:
    match = re.search(r"Action:\s*(\{.*\})", content, re.S)
    if not match:
        return None
    raw = match.group(1).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or "tool" not in parsed:
        return None
    args = parsed.get("args") or {}
    if not isinstance(args, dict):
        return None
    return {"tool": parsed["tool"], "args": args}

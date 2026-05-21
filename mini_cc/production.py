from __future__ import annotations

import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, TypeVar
from uuid import uuid4

warnings.filterwarnings("ignore", message=r".*allowed_objects.*")
try:
    from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

    warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)
except Exception:
    pass

from langgraph.graph import END, StateGraph
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from mini_cc.browser_agent import BrowserAgent
from mini_cc.events import AgentEvent, EventBus
from mini_cc.gui_agent import GUIAgent
from mini_cc.mcp import MCPClient, MCPToolServer
from mini_cc.memory import LongTermMemory
from mini_cc.observability import AgentTrace, CostTracker
from mini_cc.production_state import AgentAssignment, AgentResult, AgentRole, ProductionState, SharedState
from mini_cc.rag import CodebaseRAG
from mini_cc.state import TaskItem
from mini_cc.task_queue import InMemoryTaskQueue
from mini_cc.tools import ToolRegistry

T = TypeVar("T", bound=BaseModel)


class RouterDecision(BaseModel):
    thought: str
    assignments: list[AgentAssignment]
    done: bool = False
    final_answer: str | None = None


class ProductionAgentRuntime:
    """Mini Claude Code v3 production-style multi-agent runtime."""

    def __init__(
        self,
        workspace: str,
        model: str,
        api_key: str,
        base_url: str | None = None,
        max_iterations: int = 12,
        parallelism: int = 3,
        event_sink=None,
    ):
        self.workspace = workspace
        self.model = model
        self.max_iterations = max_iterations
        self.parallelism = parallelism
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.events = EventBus()
        self.tools = ToolRegistry(workspace)
        self.mcp_server = MCPToolServer(self.tools)
        self.mcp_client = MCPClient(self.mcp_server)
        self.rag = CodebaseRAG(workspace)
        self.memory = LongTermMemory(workspace)
        self.browser = BrowserAgent()
        self.gui = GUIAgent()
        self.queue = InMemoryTaskQueue()
        self.costs = CostTracker()
        self.event_sink = event_sink or (lambda event: None)
        self.events.subscribe("*", self._emit_external)
        self.graph = self._build_graph()

    def run(self, goal: str) -> ProductionState:
        run_id = uuid4().hex
        self.trace = AgentTrace(self.workspace, run_id)
        self.events.subscribe("*", self.trace.write)
        indexed = self.rag.index()
        self.events.emit("rag.indexed", "rag", run_id, documents=indexed)
        shared = SharedState(
            run_id=run_id,
            user_goal=goal,
            workspace=self.workspace,
            max_iterations=self.max_iterations,
        )
        state: ProductionState = {"shared": shared, "route": "router"}
        return self.graph.invoke(state, config={"recursion_limit": self.max_iterations + 10})

    def _build_graph(self):
        graph = StateGraph(ProductionState)
        graph.add_node("router", self._router_node)
        graph.add_node("parallel_agents", self._parallel_agents_node)
        graph.add_node("integrator", self._integrator_node)
        graph.set_entry_point("router")
        graph.add_conditional_edges("router", self._route_after_router, {"agents": "parallel_agents", "done": END})
        graph.add_edge("parallel_agents", "integrator")
        graph.add_conditional_edges("integrator", self._route_after_integrator, {"router": "router", "done": END})
        return graph.compile()

    def _route_after_router(self, state: ProductionState) -> str:
        return "done" if state["shared"].status != "running" else "agents"

    def _route_after_integrator(self, state: ProductionState) -> str:
        return "done" if state["shared"].status != "running" else "router"

    def _router_node(self, state: ProductionState) -> ProductionState:
        shared = state["shared"]
        if shared.iteration >= shared.max_iterations:
            shared.status = "error"
            shared.final_answer = "Reached v3 max iterations before completion."
            return {"shared": shared, "route": "done"}

        self.events.emit("agent.started", "router", shared.run_id, iteration=shared.iteration + 1)
        decision = self._route(shared)
        self.events.emit(
            "agent.routed",
            "router",
            shared.run_id,
            assignments=[assignment.model_dump() for assignment in decision.assignments],
            done=decision.done,
        )
        if decision.done:
            shared.status = "done"
            shared.final_answer = decision.final_answer or self._final_answer(shared)
            self.memory.remember_task(shared.user_goal, shared.final_answer, True)
            self.events.emit("agent.finished", "router", shared.run_id, final_answer=shared.final_answer)
            return {"shared": shared, "route": "done"}

        shared.assignments = decision.assignments or self._default_assignments(shared)
        for assignment in shared.assignments:
            self.queue.push(assignment)
            self.events.emit("task.queued", "queue", shared.run_id, assignment=assignment.model_dump())
        shared.iteration += 1
        return {"shared": shared, "route": "agents"}

    def _parallel_agents_node(self, state: ProductionState) -> ProductionState:
        shared = state["shared"]
        items = self.queue.drain()
        if not items:
            shared.assignments = self._default_assignments(shared)
            for assignment in shared.assignments:
                items.append(self.queue.push(assignment))
            items = self.queue.drain()

        results: list[AgentResult] = []
        with ThreadPoolExecutor(max_workers=max(1, self.parallelism)) as executor:
            futures = [executor.submit(self._run_assignment, shared, item.assignment) for item in items]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                self.events.emit("agent.finished", result.role, shared.run_id, result=result.model_dump())
        shared.results.extend(results)
        return {"shared": shared, "route": "integrator"}

    def _integrator_node(self, state: ProductionState) -> ProductionState:
        shared = state["shared"]
        latest = shared.results[-self.parallelism :]
        if latest:
            shared.todos = _merge_todos(shared.todos, latest)
            for result in latest:
                for assignment in result.next_actions:
                    self.queue.push(assignment)

        if _looks_complete(shared):
            shared.status = "done"
            shared.final_answer = self._final_answer(shared)
            self.memory.remember_task(shared.user_goal, shared.final_answer, True)
            self.events.emit("agent.finished", "integrator", shared.run_id, final_answer=shared.final_answer)
            return {"shared": shared, "route": "done"}
        return {"shared": shared, "route": "router"}

    def _run_assignment(self, shared: SharedState, assignment: AgentAssignment) -> AgentResult:
        self.events.emit("agent.started", assignment.role, shared.run_id, task=assignment.task)
        if assignment.role == "browser":
            return self._browser_result(assignment)
        if assignment.role == "gui":
            return AgentResult(role="gui", summary=self.gui.screenshot().message, success=False)
        if assignment.role == "coding":
            return self._coding_result(shared, assignment)
        if assignment.role == "test":
            return self._test_result(assignment)
        if assignment.role == "reviewer":
            return self._review_result(assignment)
        if assignment.role == "debug":
            return self._debug_result(shared, assignment)
        return self._llm_agent_result(shared, assignment)

    def _route(self, shared: SharedState) -> RouterDecision:
        context = self._runtime_context(shared)
        prompt = ROUTER_PROMPT.format(context=context)
        parsed = self._json_llm(prompt, RouterDecision)
        if parsed:
            return parsed
        if shared.results and any(result.role == "reviewer" and result.success for result in shared.results[-3:]):
            return RouterDecision(thought="Fallback sees reviewer success.", assignments=[], done=True)
        return RouterDecision(thought="Fallback route.", assignments=self._default_assignments(shared))

    def _default_assignments(self, shared: SharedState) -> list[AgentAssignment]:
        if not shared.results:
            return [
                AgentAssignment(role="planner", task="Create a production implementation plan", parallel=False),
                AgentAssignment(role="coding", task="Inspect codebase and identify implementation targets", parallel=True),
                AgentAssignment(role="test", task="Identify available verification commands", parallel=True),
            ]
        return [
            AgentAssignment(role="coding", task="Implement the next safe production-agent improvement", parallel=True),
            AgentAssignment(role="test", task="Run lightweight verification", parallel=True),
            AgentAssignment(role="reviewer", task="Review the current diff and completion evidence", parallel=True),
        ]

    def _coding_result(self, shared: SharedState, assignment: AgentAssignment) -> AgentResult:
        retrieved = self.rag.retrieve(shared.user_goal + " " + assignment.task, k=3)
        self.events.emit("rag.retrieved", "rag", shared.run_id, count=len(retrieved))
        summary = "Coding context retrieved:\n" + "\n\n".join(retrieved)
        return AgentResult(role="coding", summary=summary or "No coding context found.", success=True)

    def _test_result(self, assignment: AgentAssignment) -> AgentResult:
        self.events.emit("tool.started", "RunTestTool", "local", command="python -m compileall mini_cc")
        output = self.tools.run("RunTestTool", {"command": "python -m compileall mini_cc", "timeout_seconds": 60})
        success = "exit_code=0" in output
        self.events.emit("tool.finished" if success else "tool.failed", "RunTestTool", "local", output=output[:2000])
        return AgentResult(role="test", summary=output, success=success, evidence=[output[:1000]])

    def _review_result(self, assignment: AgentAssignment) -> AgentResult:
        diff = self.tools.run("GitDiffTool", {"path": ".", "max_chars": 8000})
        success = "exit_code=0" in diff
        return AgentResult(role="reviewer", summary=diff, success=success, evidence=[diff[:1000]])

    def _debug_result(self, shared: SharedState, assignment: AgentAssignment) -> AgentResult:
        failures = [result.summary for result in shared.results if not result.success]
        summary = "\n\n".join(failures[-3:]) or "No failures to debug."
        return AgentResult(role="debug", summary=summary, success=True)

    def _browser_result(self, assignment: AgentAssignment) -> AgentResult:
        url = assignment.context or self.browser.search_url(assignment.task)
        try:
            result = self.browser.fetch(url)
            return AgentResult(role="browser", summary=f"{result.title}\n{result.text[:2000]}", success=True)
        except Exception as exc:
            return AgentResult(role="browser", summary=str(exc), success=False)

    def _llm_agent_result(self, shared: SharedState, assignment: AgentAssignment) -> AgentResult:
        prompt = AGENT_PROMPT.format(
            role=assignment.role,
            task=assignment.task,
            context=self._runtime_context(shared),
        )
        parsed = self._json_llm(prompt, AgentResult)
        return parsed or AgentResult(role=assignment.role, summary="Agent returned invalid JSON.", success=False)

    def _json_llm(self, prompt: str, model: type[T]) -> T | None:
        self.costs.add_prompt(prompt)
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            stream=True,
        )
        chunks: list[str] = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                chunks.append(delta)
                self.events.emit("stream.token", "llm", "stream", token=delta)
        content = "".join(chunks)
        self.costs.add_completion(content)
        raw = _extract_json(content)
        if not raw:
            return None
        try:
            return model.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError, TypeError):
            return None

    def _runtime_context(self, shared: SharedState) -> str:
        payload = {
            "goal": shared.user_goal,
            "todos": [todo.model_dump() for todo in shared.todos],
            "recent_results": [result.model_dump() for result in shared.results[-5:]],
            "long_term_memory": self.memory.summary(),
            "mcp_tools": [tool.model_dump() for tool in self.mcp_client.list_tools()],
            "costs": self.costs.summary(),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _final_answer(self, shared: SharedState) -> str:
        summaries = "\n".join(f"- {result.role}: {result.summary[:500]}" for result in shared.results[-8:])
        return f"Mini Claude Code v3 completed the production-agent run.\n\n{summaries}"

    def _emit_external(self, event: AgentEvent) -> None:
        self.event_sink({"type": "v3_event", "event": event.model_dump()})


ROUTER_PROMPT = """You are the Agent Router for Mini Claude Code v3.

Route work between planner, coding, test, reviewer, debug, browser, and gui agents.
Return strict JSON:
{{
  "thought": "routing rationale",
  "assignments": [
    {{"role": "coding", "task": "inspect implementation", "context": "", "parallel": true}}
  ],
  "done": false,
  "final_answer": null
}}

Context:
{context}
"""


AGENT_PROMPT = """You are the {role} agent in Mini Claude Code v3.

Complete your assigned work and return strict JSON matching AgentResult:
{{
  "role": "{role}",
  "summary": "what you found or did",
  "success": true,
  "evidence": ["short evidence"],
  "next_actions": [],
  "tool_calls": []
}}

Task:
{task}

Shared context:
{context}
"""


def _extract_json(content: str) -> str | None:
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end <= start:
        return None
    return content[start : end + 1]


def _merge_todos(todos: list[TaskItem], results: list[AgentResult]) -> list[TaskItem]:
    if not todos:
        return [
            TaskItem(id=1, title="Plan production-agent work", status="DONE"),
            TaskItem(id=2, title="Run coding/test/review agents", status="DONE" if results else "RUNNING"),
        ]
    if results and all(result.success for result in results):
        return [todo.model_copy(update={"status": "DONE"}) for todo in todos]
    return todos


def _looks_complete(shared: SharedState) -> bool:
    if shared.iteration < 2:
        return False
    recent = shared.results[-3:]
    roles = {result.role for result in recent if result.success}
    return {"coding", "test", "reviewer"}.issubset(roles)

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from mini_cc.tools import ToolRegistry


class MCPToolDescriptor(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: str | int
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: str | int
    result: Any | None = None
    error: dict[str, Any] | None = None


class MCPToolServer:
    """In-process MCP-like tool server for discovery and dynamic calls."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def discover(self) -> list[MCPToolDescriptor]:
        return [
            MCPToolDescriptor(
                name=tool.name,
                description=tool.description,
                input_schema=tool.input_model.model_json_schema(),
            )
            for tool in self.registry.tools.values()
        ]

    def handle(self, raw_request: str) -> str:
        try:
            request = MCPRequest.model_validate(json.loads(raw_request))
            if request.method == "tools/list":
                response = MCPResponse(id=request.id, result=[tool.model_dump() for tool in self.discover()])
            elif request.method == "tools/call":
                name = request.params["name"]
                args = request.params.get("arguments", {})
                response = MCPResponse(id=request.id, result=self.registry.run(name, args))
            else:
                response = MCPResponse(id=request.id, error={"code": -32601, "message": "method not found"})
        except Exception as exc:
            response = MCPResponse(id="unknown", error={"code": -32000, "message": str(exc)})
        return response.model_dump_json()


class MCPClient:
    def __init__(self, server: MCPToolServer):
        self.server = server
        self._counter = 0

    def list_tools(self) -> list[MCPToolDescriptor]:
        response = self._send("tools/list", {})
        return [MCPToolDescriptor.model_validate(item) for item in response.result or []]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        response = self._send("tools/call", {"name": name, "arguments": arguments})
        if response.error:
            raise RuntimeError(response.error["message"])
        return str(response.result)

    def _send(self, method: str, params: dict[str, Any]) -> MCPResponse:
        self._counter += 1
        request = MCPRequest(id=self._counter, method=method, params=params)
        return MCPResponse.model_validate_json(self.server.handle(request.model_dump_json()))

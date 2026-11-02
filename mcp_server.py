"""Local MCP server exposing the incident-triage tools over stdio.

Thin wrapper only: all actual logic (simulated telemetry, runbook search)
still lives in tools.py / knowledge.py and is unchanged. This module just
re-exposes those existing LangChain tools as MCP tools, by delegating each
MCP tool call to the real tool's `.invoke()`, so graph.py's ReAct agents can
consume them via langchain-mcp-adapters instead of importing them directly
in-process.

Run as a subprocess by graph.py's MultiServerMCPClient (stdio transport);
not meant to be run standalone except for manual testing:

    python mcp_server.py
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from knowledge import query_runbooks_tool
from tools import fetch_service_logs, fetch_service_metrics

mcp = FastMCP("incident-triage-tools")


@mcp.tool()
def fetch_service_logs_mcp(service_name: str) -> str:
    """Fetch recent error-level log traces for the given service.

    Args:
        service_name: The Kubernetes service/deployment name, e.g.
            "payment-service" or "auth-service".
    """
    return fetch_service_logs.invoke({"service_name": service_name})


@mcp.tool()
def fetch_service_metrics_mcp(service_name: str) -> str:
    """Fetch current Prometheus-style metrics for the given service.

    Args:
        service_name: The Kubernetes service/deployment name, e.g.
            "payment-service" or "auth-service".
    """
    return fetch_service_metrics.invoke({"service_name": service_name})


@mcp.tool()
def query_runbooks_mcp(query: str) -> str:
    """Search internal SRE runbooks for incident diagnosis and remediation guidance.

    Args:
        query: A natural-language description of the incident/service.
    """
    return query_runbooks_tool.invoke({"query": query})


if __name__ == "__main__":
    mcp.run(transport="stdio")

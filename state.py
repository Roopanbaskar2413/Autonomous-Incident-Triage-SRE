"""Shared LangGraph state definition for the incident triage agent."""

from __future__ import annotations

from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class IncidentState(TypedDict):
    """State threaded through every node of the incident triage graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    alert_payload: dict[str, Any]
    service_name: str
    telemetry_findings: str
    runbook_guidance: str
    rca_summary: str
    proposed_remediation: dict[str, Any]
    human_approved: Optional[bool]

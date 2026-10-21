"""LangGraph pipeline for autonomous incident triage with a human approval gate."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from knowledge import query_runbooks
from state import IncidentState
from tools import execute_remediation, fetch_service_logs, fetch_service_metrics


class RCAOutput(BaseModel):
    """Structured root-cause-analysis output produced by the LLM."""

    rca_summary: str = Field(
        description=(
            "A concise, operator-facing root cause analysis: what is "
            "happening, why, and the evidence that supports the diagnosis."
        )
    )
    remediation_action: str = Field(
        description=(
            "The single remediation action to propose, e.g. "
            "'rolling_restart' or 'scale_replicas'."
        )
    )
    remediation_target: str = Field(
        description=(
            "The Kubernetes resource the action targets, e.g. "
            "'payment-service-deployment'."
        )
    )
    remediation_reason: str = Field(
        description="One-sentence justification for why this action was chosen."
    )


def _get_llm() -> ChatOllama:
    return ChatOllama(model="incident-triage-llama", temperature=0)


def triage_node(state: IncidentState) -> dict[str, Any]:
    """Parse the incoming alert payload and identify the affected service."""
    alert = state["alert_payload"]
    service_name = alert.get("service_name") or alert.get("service", "unknown-service")

    summary = (
        f"P1 alert received: severity={alert.get('severity', 'unknown')}, "
        f"service={service_name}, title={alert.get('title', 'n/a')!r}, "
        f"description={alert.get('description', 'n/a')!r}."
    )
    return {
        "service_name": service_name,
        "messages": [HumanMessage(content=summary)],
    }


def telemetry_inspector_node(state: IncidentState) -> dict[str, Any]:
    """Pull simulated logs and metrics for the affected service."""
    service_name = state["service_name"]

    logs = fetch_service_logs.invoke({"service_name": service_name})
    metrics = fetch_service_metrics.invoke({"service_name": service_name})

    findings = (
        f"--- Logs for {service_name} ---\n{logs}\n"
        f"--- Metrics for {service_name} ---\n{metrics}"
    )
    return {
        "telemetry_findings": findings,
        "messages": [AIMessage(content=f"Collected telemetry for {service_name}.")],
    }


def runbook_matcher_node(state: IncidentState) -> dict[str, Any]:
    """Query the runbook knowledge base using the telemetry findings."""
    service_name = state["service_name"]
    query = (
        f"{service_name} incident diagnosis and remediation given symptoms: "
        f"{state['telemetry_findings'][:800]}"
    )
    guidance = query_runbooks(query)
    return {
        "runbook_guidance": guidance,
        "messages": [AIMessage(content="Matched incident against internal runbooks.")],
    }


def rca_synthesizer_node(state: IncidentState) -> dict[str, Any]:
    """Synthesize a structured RCA and proposed remediation via the LLM."""
    llm = _get_llm().with_structured_output(RCAOutput)

    prompt = (
        "You are an SRE copilot performing root cause analysis for a "
        "production incident. Using the telemetry findings and runbook "
        "guidance below, produce a concise RCA and propose exactly one "
        "remediation action.\n\n"
        f"Service: {state['service_name']}\n\n"
        f"Telemetry findings:\n{state['telemetry_findings']}\n\n"
        f"Runbook guidance:\n{state['runbook_guidance']}\n"
    )

    result: RCAOutput = llm.invoke([HumanMessage(content=prompt)])

    proposed_remediation = {
        "action": result.remediation_action,
        "target": result.remediation_target,
        "reason": result.remediation_reason,
    }
    return {
        "rca_summary": result.rca_summary,
        "proposed_remediation": proposed_remediation,
        "human_approved": None,
        "messages": [AIMessage(content=f"RCA complete: {result.rca_summary}")],
    }


def remediation_node(state: IncidentState) -> dict[str, Any]:
    """Execute the proposed remediation, but only if a human approved it.

    This node runs after the graph resumes from the `interrupt_before`
    breakpoint. `main.py` is responsible for setting `human_approved` on
    the checkpointed state before resuming execution.
    """
    if not state.get("human_approved"):
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Remediation was NOT executed: operator did not "
                        "approve the proposed action."
                    )
                )
            ],
        }

    remediation = state["proposed_remediation"]
    result = execute_remediation.invoke(
        {"action": remediation["action"], "target": remediation["target"]}
    )
    return {"messages": [AIMessage(content=result)]}


def build_graph() -> Any:
    """Build and compile the incident triage graph with a HITL breakpoint."""
    graph = StateGraph(IncidentState)

    graph.add_node("triage", triage_node)
    graph.add_node("telemetry_inspector", telemetry_inspector_node)
    graph.add_node("runbook_matcher", runbook_matcher_node)
    graph.add_node("rca_synthesizer", rca_synthesizer_node)
    graph.add_node("remediation", remediation_node)

    graph.add_edge(START, "triage")
    graph.add_edge("triage", "telemetry_inspector")
    graph.add_edge("telemetry_inspector", "runbook_matcher")
    graph.add_edge("runbook_matcher", "rca_synthesizer")
    graph.add_edge("rca_synthesizer", "remediation")
    graph.add_edge("remediation", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer, interrupt_before=["remediation"])


def proposed_remediation_to_command(proposed_remediation: dict[str, Any]) -> str:
    """Render the proposed remediation dict as a human-readable command line."""
    return json.dumps(proposed_remediation, indent=2)

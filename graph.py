"""LangGraph multi-agent pipeline for autonomous incident triage.

A supervisor agent routes between two specialist investigator agents
(telemetry, runbooks) until enough evidence is gathered, then an RCA
synthesizer agent produces the root cause analysis and a single proposed
remediation, which a human must approve before execution.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from knowledge import query_runbooks_tool
from state import IncidentState
from tools import execute_remediation, fetch_service_logs, fetch_service_metrics

MAX_SUPERVISOR_ITERATIONS = 4

TELEMETRY_AGENT_PROMPT = (
    "You are a telemetry investigation agent on an SRE incident response "
    "team. Given a production incident, use the fetch_service_logs and "
    "fetch_service_metrics tools to gather evidence for the affected "
    "service. Call whichever tools you need, then respond with a concise, "
    "factual summary of what the logs and metrics show. Do not propose a "
    "remediation yourself; that is another agent's job."
)

RUNBOOK_AGENT_PROMPT = (
    "You are a runbook research agent on an SRE incident response team. "
    "Given a production incident and the telemetry evidence gathered so "
    "far, use the query_runbooks_tool tool to search internal SRE runbooks "
    "for matching diagnosis and remediation guidance. Respond with a "
    "concise summary of the relevant guidance found."
)


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


class SupervisorDecision(BaseModel):
    """The supervisor's routing decision for which specialist acts next."""

    next: Literal["telemetry", "runbook", "synthesize"] = Field(
        description=(
            "Which specialist agent to invoke next: 'telemetry' to gather "
            "logs/metrics, 'runbook' to search internal runbooks, or "
            "'synthesize' once there is enough evidence (telemetry findings "
            "AND runbook guidance) to produce a root cause analysis."
        )
    )
    reasoning: str = Field(description="One sentence explaining the choice.")


def _get_tool_llm() -> ChatOllama:
    """Tool-calling capable model, used by the supervisor and specialist agents."""
    return ChatOllama(model="llama3.2", temperature=0)


def _get_rca_llm() -> ChatOllama:
    """Fine-tuned model used only for the final structured RCA synthesis."""
    return ChatOllama(model="incident-triage-llama", temperature=0)


def _build_telemetry_agent() -> Any:
    return create_react_agent(
        _get_tool_llm(),
        tools=[fetch_service_logs, fetch_service_metrics],
        prompt=TELEMETRY_AGENT_PROMPT,
    )


def _build_runbook_agent() -> Any:
    return create_react_agent(
        _get_tool_llm(),
        tools=[query_runbooks_tool],
        prompt=RUNBOOK_AGENT_PROMPT,
    )


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
        "supervisor_iterations": 0,
        "messages": [HumanMessage(content=summary)],
    }


def supervisor_node(state: IncidentState) -> dict[str, Any]:
    """LLM-driven router that decides which specialist agent investigates next."""
    iterations = state.get("supervisor_iterations", 0) + 1
    have_telemetry = bool(state.get("telemetry_findings"))
    have_runbook = bool(state.get("runbook_guidance"))

    if have_telemetry and have_runbook:
        next_step = "synthesize"
        reasoning = "Telemetry and runbook evidence have both been gathered."
    elif iterations > MAX_SUPERVISOR_ITERATIONS:
        next_step = "synthesize"
        reasoning = "Iteration limit reached; proceeding with evidence gathered so far."
    else:
        llm = _get_tool_llm().with_structured_output(SupervisorDecision)
        prompt = (
            "You are the supervisor of an SRE incident-response team "
            "investigating a production incident. Decide which specialist "
            "agent should act next so the team has enough evidence to "
            "diagnose the incident.\n\n"
            f"Service: {state['service_name']}\n"
            f"Telemetry findings gathered so far: {'yes' if have_telemetry else 'no'}\n"
            f"Runbook guidance gathered so far: {'yes' if have_runbook else 'no'}\n"
        )
        decision: SupervisorDecision = llm.invoke([HumanMessage(content=prompt)])
        next_step, reasoning = decision.next, decision.reasoning

    return {
        "next_agent": next_step,
        "supervisor_iterations": iterations,
        "messages": [AIMessage(content=f"[supervisor] -> {next_step}: {reasoning}")],
    }


def _route_from_supervisor(state: IncidentState) -> str:
    return {
        "telemetry": "telemetry_agent",
        "runbook": "runbook_agent",
        "synthesize": "rca_synthesizer",
    }.get(state.get("next_agent", "synthesize"), "rca_synthesizer")


def telemetry_agent_node(state: IncidentState) -> dict[str, Any]:
    """Specialist agent: investigates logs/metrics for the affected service."""
    alert = state["alert_payload"]
    service_name = state["service_name"]
    task = (
        f"Investigate service '{service_name}'. Alert: "
        f"{alert.get('title', 'n/a')} - {alert.get('description', 'n/a')}"
    )
    agent = _build_telemetry_agent()
    result = agent.invoke({"messages": [HumanMessage(content=task)]})
    findings = result["messages"][-1].content
    return {
        "telemetry_findings": findings,
        "messages": [AIMessage(content=f"[telemetry-agent] {findings}")],
    }


def runbook_agent_node(state: IncidentState) -> dict[str, Any]:
    """Specialist agent: searches internal runbooks for matching guidance."""
    service_name = state["service_name"]
    task = (
        f"Service '{service_name}' incident. Telemetry findings so far:\n"
        f"{state.get('telemetry_findings', '(none gathered yet)')}\n\n"
        "Find matching runbook guidance for diagnosis and remediation."
    )
    agent = _build_runbook_agent()
    result = agent.invoke({"messages": [HumanMessage(content=task)]})
    guidance = result["messages"][-1].content
    return {
        "runbook_guidance": guidance,
        "messages": [AIMessage(content=f"[runbook-agent] {guidance}")],
    }


def rca_synthesizer_node(state: IncidentState) -> dict[str, Any]:
    """Synthesize a structured RCA and proposed remediation via the LLM."""
    llm = _get_rca_llm().with_structured_output(RCAOutput)

    prompt = (
        "You are an SRE copilot performing root cause analysis for a "
        "production incident. Using the telemetry findings and runbook "
        "guidance below, produce a concise RCA and propose exactly one "
        "remediation action.\n\n"
        f"Service: {state['service_name']}\n\n"
        f"Telemetry findings:\n{state.get('telemetry_findings', '(none gathered)')}\n\n"
        f"Runbook guidance:\n{state.get('runbook_guidance', '(none gathered)')}\n"
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
    """Build and compile the multi-agent incident triage graph with a HITL breakpoint."""
    graph = StateGraph(IncidentState)

    graph.add_node("triage", triage_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("telemetry_agent", telemetry_agent_node)
    graph.add_node("runbook_agent", runbook_agent_node)
    graph.add_node("rca_synthesizer", rca_synthesizer_node)
    graph.add_node("remediation", remediation_node)

    graph.add_edge(START, "triage")
    graph.add_edge("triage", "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {
            "telemetry_agent": "telemetry_agent",
            "runbook_agent": "runbook_agent",
            "rca_synthesizer": "rca_synthesizer",
        },
    )
    graph.add_edge("telemetry_agent", "supervisor")
    graph.add_edge("runbook_agent", "supervisor")
    graph.add_edge("rca_synthesizer", "remediation")
    graph.add_edge("remediation", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer, interrupt_before=["remediation"])


def proposed_remediation_to_command(proposed_remediation: dict[str, Any]) -> str:
    """Render the proposed remediation dict as a human-readable command line."""
    return json.dumps(proposed_remediation, indent=2)

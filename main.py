"""Entry point: run the incident triage graph up to the HITL approval gate,
show the operator the RCA and proposed remediation, then resume on approval.
"""

from __future__ import annotations

import sys
from typing import Any

from dotenv import load_dotenv

from graph import build_graph, proposed_remediation_to_command
from state import IncidentState

SAMPLE_ALERT: dict[str, Any] = {
    "service_name": "payment-service",
    "severity": "P1",
    "title": "High 5xx rate and elevated latency on payment-service",
    "description": (
        "Alertmanager: payment-service HTTP 504 rate above 15% for 5m. "
        "p99 latency above 5s. Suspected connection pool exhaustion."
    ),
}


def _print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    load_dotenv()

    app = build_graph()
    config = {"configurable": {"thread_id": "incident-001"}}

    initial_state: IncidentState = {
        "messages": [],
        "alert_payload": SAMPLE_ALERT,
        "service_name": "",
        "telemetry_findings": "",
        "runbook_guidance": "",
        "rca_summary": "",
        "proposed_remediation": {},
        "human_approved": None,
    }

    _print_header("INCIDENT ALERT RECEIVED")
    print(f"Service:     {SAMPLE_ALERT['service_name']}")
    print(f"Severity:    {SAMPLE_ALERT['severity']}")
    print(f"Title:       {SAMPLE_ALERT['title']}")
    print(f"Description: {SAMPLE_ALERT['description']}")

    try:
        app.invoke(initial_state, config=config)
    except Exception as exc:  # noqa: BLE001 - surface any pipeline failure clearly
        print(f"\nIncident triage pipeline failed before the approval gate: {exc}")
        sys.exit(1)

    snapshot = app.get_state(config)
    state = snapshot.values

    _print_header("ROOT CAUSE ANALYSIS")
    print(state.get("rca_summary", "(no RCA produced)"))

    _print_header("PROPOSED REMEDIATION")
    print(proposed_remediation_to_command(state.get("proposed_remediation", {})))

    _print_header("HUMAN APPROVAL REQUIRED")
    answer = input("Approve remediation? (yes/no): ").strip().lower()
    approved = answer in {"y", "yes"}

    app.update_state(config, {"human_approved": approved})

    try:
        app.invoke(None, config=config)
    except Exception as exc:  # noqa: BLE001
        print(f"\nRemediation step failed: {exc}")
        sys.exit(1)

    final_snapshot = app.get_state(config)
    final_messages = final_snapshot.values.get("messages", [])

    _print_header("OUTCOME")
    if final_messages:
        print(final_messages[-1].content)
    else:
        print("No outcome message produced.")


if __name__ == "__main__":
    main()

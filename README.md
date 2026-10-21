# Autonomous Incident Triage (SRE)

An autonomous SRE copilot that triages production incidents end-to-end, with a human-in-the-loop approval gate before any remediation is executed.

## Pipeline

Built with [LangGraph](https://github.com/langchain-ai/langgraph), the agent runs as a state machine:

1. **Triage** — parses the incoming alert payload and identifies the affected service.
2. **Telemetry Inspector** — pulls recent logs and metrics for the service.
3. **Runbook Matcher** — retrieves relevant guidance from internal SRE runbooks using a local vector index (LlamaIndex + HuggingFace embeddings, no external API required).
4. **RCA Synthesizer** — uses an LLM (local via [Ollama](https://ollama.com)) to produce a root cause analysis and propose exactly one remediation action.
5. **Human Approval Gate** — the graph pauses and presents the RCA and proposed remediation to an operator for approval/rejection.
6. **Remediation** — executes the approved action (simulated; no real cluster is touched) and reports the outcome.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pull a local model via Ollama (e.g. `ollama pull llama3.2`) and make sure the Ollama server is running.

## Run

```bash
python main.py
```

A sample `payment-service` P1 alert is triaged end-to-end; you'll be prompted to approve or reject the proposed remediation before it executes.

## Project layout

- `main.py` — entry point; runs the graph up to the approval gate, then resumes on operator input.
- `graph.py` — LangGraph pipeline definition and node logic.
- `state.py` — shared graph state schema.
- `tools.py` — simulated telemetry fetch and remediation execution tools.
- `knowledge.py` — local runbook knowledge base (LlamaIndex retrieval).
- `runbooks/` — source runbook documents indexed for retrieval.

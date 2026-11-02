# Autonomous Incident Triage (SRE)

An autonomous SRE copilot that triages production incidents end-to-end, with a human-in-the-loop approval gate before any remediation is executed.

## Architecture: multi-agent orchestration

Built with [LangGraph](https://github.com/langchain-ai/langgraph), as a supervisor coordinating specialist agents rather than a fixed linear pipeline:

1. **Triage** — parses the incoming alert payload and identifies the affected service.
2. **Supervisor** — an LLM router that looks at what evidence has been gathered so far and decides which specialist investigates next (`telemetry`, `runbook`, or `synthesize` once there's enough to proceed). It re-runs after each specialist returns, so it can loop back for more evidence, skip a specialist, or stop early — capped at a few iterations as a safety net against indecisive routing.
3. **Telemetry Agent** — a ReAct agent (via `langgraph.prebuilt.create_react_agent`) that decides for itself which of `fetch_service_logs_mcp` / `fetch_service_metrics_mcp` to call for the affected service, then summarizes what it found.
4. **Runbook Agent** — a ReAct agent that searches internal SRE runbooks (local vector index — LlamaIndex + HuggingFace embeddings, no external API required) for guidance matching the incident.
5. **RCA Synthesizer** — once the supervisor has enough evidence, this node uses an LLM (local via [Ollama](https://ollama.com)) to produce a root cause analysis and propose exactly one remediation action. The model, `incident-triage-llama`, is a LoRA fine-tune of `Llama-3.2-3B-Instruct` trained to reliably emit the exact structured schema (e.g. short `remediation_action` slugs like `rolling_restart` instead of prose) — see [finetune/README.md](finetune/README.md) for how it was built and how to reproduce it. The two specialist agents and the supervisor run on the stock `llama3.2` model instead, since they need native tool-calling rather than the fine-tune's structured-output specialization.
6. **Human Approval Gate** — the graph pauses and presents the RCA and proposed remediation to an operator for approval/rejection.
7. **Remediation** — executes the approved action (simulated; no real cluster is touched) and reports the outcome.

### Observability: LangSmith tracing

When `LANGCHAIN_TRACING_V2=true` is set (see `.env.example`), LangGraph/LangChain automatically trace every run to [LangSmith](https://smith.langchain.com) — no code changes required. Each run shows the full multi-agent execution as a tree: the triage node, every supervisor routing decision, each specialist agent's tool calls, and the RCA synthesizer, with per-node latency and token cost broken out individually.

### Tool serving: MCP (Model Context Protocol)

The Telemetry and Runbook agents' tools are no longer imported in-process — they're served over MCP by `mcp_server.py`, a thin wrapper that re-exposes the real `fetch_service_logs` / `fetch_service_metrics` (`tools.py`) and `query_runbooks_tool` (`knowledge.py`) logic as `fetch_service_logs_mcp`, `fetch_service_metrics_mcp`, and `query_runbooks_mcp` MCP tools, unchanged underneath. `graph.py` connects to it via `langchain-mcp-adapters`' `MultiServerMCPClient` over stdio, spawning `mcp_server.py` as a subprocess.

Loading the MCP tool list is async, so `init_mcp_tools()` must be awaited once at startup, before `build_graph()` — this is why `main.py`'s entrypoint is now `async def main()` run via a single `asyncio.run(main())`, with the graph itself driven through `app.ainvoke()` / `app.aget_state()` / `app.aupdate_state()` instead of the sync equivalents. The supervisor, RCA synthesizer, and remediation nodes don't touch MCP and stay plain sync functions; LangGraph runs sync and async nodes together in the same graph without issue.

One real tradeoff from this change: per `langchain-mcp-adapters`, a new MCP session (and, with the stdio transport, a new `mcp_server.py` subprocess) is spawned for every individual tool call, not reused across a run. In practice that means each telemetry/runbook tool call reloads the local embedding model used by `knowledge.py` on import — a few hundred milliseconds of extra startup per call, compared to the old in-process version. Functionally correct, just not free.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Build (or rebuild) the fine-tuned `incident-triage-llama` Ollama model by following [finetune/README.md](finetune/README.md), and make sure the Ollama server is running.

## Run

```bash
python main.py
```

A sample `payment-service` P1 alert is triaged end-to-end; you'll be prompted to approve or reject the proposed remediation before it executes.

## Project layout

- `main.py` — async entry point; loads MCP tools, runs the graph up to the approval gate, then resumes on operator input.
- `graph.py` — LangGraph multi-agent graph: supervisor, specialist agents (consuming tools over MCP), and node logic.
- `mcp_server.py` — local MCP server (stdio) exposing `tools.py` / `knowledge.py`'s tools; run as a subprocess by `graph.py`.
- `state.py` — shared graph state schema.
- `tools.py` — simulated telemetry fetch and remediation execution tools.
- `knowledge.py` — local runbook knowledge base (LlamaIndex retrieval).
- `runbooks/` — source runbook documents indexed for retrieval.

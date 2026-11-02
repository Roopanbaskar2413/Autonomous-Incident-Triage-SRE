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

## Deployment

The app is containerized and deployable to a local Kubernetes cluster (minikube). No cloud, no cost — everything runs on the host machine.

### Design decision: where does Ollama run?

**Ollama stays external, running on the host**, not as an in-cluster pod. The app pod reaches it at `http://host.minikube.internal:11434` — a DNS name minikube provides specifically so pods can reach the host machine, verified directly (a throwaway pod resolved it and got a real response from the host's `ollama serve`).

Why not run Ollama in-cluster too:

1. **The weights already exist on disk.** `llama3.2` + `incident-triage-llama` are several GB, already pulled into `~/.ollama`. Baking them into an image means a multi-GB layer; having an in-cluster Ollama pod `ollama pull` them at startup means re-downloading a model that's sitting right there on the host. Neither buys anything.
2. **A single-node local cluster gives none of the reasons you'd normally containerize a model server.** You containerize model-serving in production for independent scaling, GPU node scheduling, or multi-tenant isolation — none of which exist with one node. There's nothing to schedule it apart from, nothing to isolate it from, nothing to scale independently.
3. **It mirrors real practice**: what you deploy to Kubernetes is the application; it calls out to an inference backend (hosted API, internal GPU cluster, or here, a host-resident Ollama) as an external dependency over the network, rather than bundling infrastructure that already runs fine as a host service into the app's own deployment unit.

The honest counter-case: if this needed to show "model-serving containerized too" (e.g. for a multi-node setup, or to let other in-cluster services reach the model without a host dependency), the defensible version wouldn't bake weights into an image either — it'd be a separate Ollama Deployment + Service with the host's `~/.ollama` mounted in via `hostPath` (reusing the already-pulled blobs), reached via in-cluster DNS instead of `host.minikube.internal`. Not done here because it adds real complexity (hostPath path-mapping through Colima's VM) for no benefit at this scale — but it's a contained, specific change if that constraint ever shows up.

**MCP needs no extra container.** `mcp_server.py` is spawned by `graph.py` as a stdio subprocess of the same Python process — a child process on a shared filesystem, which works identically inside one container. One Dockerfile, one pod.

**No Service manifest.** `main.py` is a one-shot CLI tool (one triage pass, then exit), not an HTTP server — nothing in the pod listens on a port, so there's no backend for a Service to route to. A Service here would be decorative.

One real code change this required: `graph.py`'s Ollama client now reads its base URL from `OLLAMA_BASE_URL` (defaulting to `http://localhost:11434`, so the plain local workflow above is unaffected), overridden to `http://host.minikube.internal:11434` in `k8s/deployment.yaml`.

### Running it

```bash
# one-time local toolchain (Colima for the Docker runtime + minikube + kubectl)
brew install colima docker minikube kubectl
colima start --cpu 4 --memory 6 --disk 30
minikube start --driver=docker --cpus=3 --memory=4096

# build the image and load it directly into minikube's node (no registry)
docker build -t incident-triage-agent:latest .
minikube image load incident-triage-agent:latest

# deploy
kubectl apply -f k8s/deployment.yaml
kubectl get pods   # wait for Running

# trigger a real triage run inside the pod. main.py needs interactive stdin
# for the human-approval gate; `kubectl exec -i` (no -t) accepts piped
# stdin non-interactively, same pattern used to test this locally.
POD=$(kubectl get pod -l app=incident-triage-agent -o jsonpath='{.items[0].metadata.name}')
echo yes | kubectl exec -i "$POD" -- python main.py
```

Verified by actually doing all of the above: cluster came up, `host.minikube.internal` was confirmed (via a throwaway test pod) to resolve to the host and reach `ollama serve`, the image built (3.52GB — mostly `torch`/CUDA deps pulled in transitively by `llama-index-embeddings-huggingface`) and loaded into minikube, and the Deployment reached `Running`.

**What actually broke on the first attempt, and the fix:** the first deploy used a 1Gi memory limit and the pod got `OOMKilled` (exit 137) partway through a run. Cause: each MCP tool call spawns its own `mcp_server.py` subprocess (per `langchain-mcp-adapters`' one-session-per-call behavior — see the MCP section above), and each of those independently imports `torch`/`sentence-transformers`/the embedding model; the telemetry agent calls two tools concurrently, so two such subprocesses can be resident in the container at once. Raised the limit to 3Gi (`k8s/deployment.yaml`) and reran — confirmed `Restart Count: 0` afterward, no further OOM.

**Real run output against the deployed pod** (approve path):

```
========================================================================
INCIDENT ALERT RECEIVED
========================================================================
Service:     payment-service
Severity:    P1
Title:       High 5xx rate and elevated latency on payment-service
Description: Alertmanager: payment-service HTTP 504 rate above 15% for 5m. p99 latency above 5s. Suspected connection pool exhaustion.

========================================================================
ROOT CAUSE ANALYSIS
========================================================================
payment-service is experiencing connection pool exhaustion due to high requests per second and elevated latency, with the connection pool utilization at 99.4%.

========================================================================
PROPOSED REMEDIATION
========================================================================
{
  "action": "rolling_restart",
  "target": "payment-service-deployment",
  "reason": "Clear leaked connections and release pool slots with a rolling restart of the payment-service-deployment."
}

========================================================================
HUMAN APPROVAL REQUIRED
========================================================================
Approve remediation? (yes/no):
========================================================================
OUTCOME
========================================================================
[2026-11-02T21:24:22.162802+00:00] Executed action='rolling_restart' on target='payment-service-deployment'. Kubernetes API acknowledged the request; rollout status: in-progress. This was a simulated execution for demo purposes.
```

The reject path (`echo no | kubectl exec -i "$POD" -- python main.py`) was also verified — correctly skips remediation, no restart.

## Project layout

- `main.py` — async entry point; loads MCP tools, runs the graph up to the approval gate, then resumes on operator input.
- `graph.py` — LangGraph multi-agent graph: supervisor, specialist agents (consuming tools over MCP), and node logic.
- `mcp_server.py` — local MCP server (stdio) exposing `tools.py` / `knowledge.py`'s tools; run as a subprocess by `graph.py`.
- `state.py` — shared graph state schema.
- `tools.py` — simulated telemetry fetch and remediation execution tools.
- `knowledge.py` — local runbook knowledge base (LlamaIndex retrieval).
- `runbooks/` — source runbook documents indexed for retrieval.
- `Dockerfile`, `.dockerignore` — container image for the app (not Ollama; see Deployment section).
- `k8s/deployment.yaml` — Kubernetes Deployment for the app pod.

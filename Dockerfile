FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py graph.py state.py tools.py knowledge.py mcp_server.py ./
COPY runbooks/ ./runbooks/

# This app is a one-shot CLI agent (main.py runs one triage pass and exits),
# not a long-running HTTP service, so there's no server process to bind a
# port to. The container stays alive so a run can be triggered on demand via
# `kubectl exec` - see README.md's Deployment section for why, and for how
# the human-approval gate is fed non-interactively for that case.
CMD ["sleep", "infinity"]

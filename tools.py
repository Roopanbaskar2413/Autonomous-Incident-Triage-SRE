"""Simulated telemetry and remediation tools exposed to the agent via LangChain.

These tools stand in for real integrations (Prometheus, Kubernetes API,
log aggregators). They return deterministic, realistic-looking payloads so
the graph can be exercised end-to-end without live infrastructure.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from langchain_core.tools import tool

# Canned telemetry keyed by service name. Any service not listed here falls
# back to a generic "healthy-ish but degraded" profile so the tool never
# raises on an unknown name.
_LOG_TRACES: dict[str, str] = {
    "payment-service": (
        "2026-10-21T09:14:02.331Z ERROR payment-service.worker-7 "
        "PostgresConnectionPoolTimeout: timeout after 30000ms waiting for "
        "connection from pool (size=50, overflow=10, checked_out=60)\n"
        "  at ConnectionPool.acquire (pg_pool.py:142)\n"
        "  at PaymentRepository.authorize_transaction (payment_repository.py:88)\n"
        "  at AuthorizeHandler.handle (authorize_handler.py:41)\n"
        "2026-10-21T09:14:02.890Z ERROR payment-service.worker-3 "
        "TimeoutError: QueuePool limit of size 50 overflow 10 reached, "
        "connection timed out, timeout 30\n"
        "  at QueuePool._do_get (queue_pool.py:211)\n"
        "  at PaymentRepository.capture_transaction (payment_repository.py:103)\n"
        "2026-10-21T09:14:03.015Z WARN  payment-service.worker-1 "
        "slow query detected: UPDATE ledger_entries SET ... duration=28410ms "
        "(idle_in_transaction=true)\n"
    ),
    "auth-service": (
        "2026-10-21T09:12:47.102Z ERROR auth-service.pod-2 "
        "TokenValidationTimeout: redis client timed out validating token "
        "after 2000ms (namespace=session_tokens)\n"
        "  at RedisTokenCache.validate (redis_token_cache.py:76)\n"
        "  at TokenValidator.validate_session (token_validator.py:29)\n"
        "2026-10-21T09:12:47.540Z WARN  auth-service.pod-1 "
        "cache eviction observed: evicted_keys=184 policy=allkeys-lru "
        "used_memory=0.98*maxmemory\n"
        "2026-10-21T09:12:48.012Z ERROR auth-service.pod-2 "
        "EvictionPolicyError: key 'session:9f3a...' evicted mid-validation, "
        "falling back to primary token store\n"
    ),
}

_METRICS: dict[str, dict[str, float | int | str]] = {
    "payment-service": {
        "service": "payment-service",
        "http_5xx_rate_pct": 18.7,
        "latency_p50_ms": 142,
        "latency_p99_ms": 6240,
        "connection_pool_utilization_pct": 99.4,
        "requests_per_sec": 312,
        "sampled_at": "2026-10-21T09:14:05Z",
    },
    "auth-service": {
        "service": "auth-service",
        "http_5xx_rate_pct": 6.3,
        "latency_p50_ms": 48,
        "latency_p99_ms": 2130,
        "cache_hit_rate_pct": 54.2,
        "redis_used_memory_pct": 97.8,
        "replica_count": 2,
        "requests_per_sec": 540,
        "sampled_at": "2026-10-21T09:12:50Z",
    },
}


@tool
def fetch_service_logs(service_name: str) -> str:
    """Fetch recent error-level log traces for the given service.

    Args:
        service_name: The Kubernetes service/deployment name, e.g.
            "payment-service" or "auth-service".

    Returns:
        A string containing recent stack traces and log lines for the
        service. Falls back to a generic trace if the service is unknown.
    """
    return _LOG_TRACES.get(
        service_name,
        f"{datetime.now(timezone.utc).isoformat()} WARN  {service_name} "
        "no recent error-level traces found; service reporting nominal "
        "health in the last observation window.\n",
    )


@tool
def fetch_service_metrics(service_name: str) -> str:
    """Fetch current Prometheus-style metrics for the given service.

    Args:
        service_name: The Kubernetes service/deployment name, e.g.
            "payment-service" or "auth-service".

    Returns:
        A JSON-encoded string with error rate, latency, and resource
        utilization metrics for the service.
    """
    metrics = _METRICS.get(
        service_name,
        {
            "service": service_name,
            "http_5xx_rate_pct": 0.5,
            "latency_p50_ms": 80,
            "latency_p99_ms": 220,
            "connection_pool_utilization_pct": 22.0,
            "requests_per_sec": 150,
            "sampled_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return json.dumps(metrics, indent=2)


@tool
def execute_remediation(action: str, target: str) -> str:
    """Execute a remediation action against a Kubernetes resource.

    This is a simulated execution: it does not touch a real cluster. In a
    production system this would shell out to `kubectl`, call the
    Kubernetes API, or invoke a runbook-automation platform, gated by the
    human-in-the-loop approval already enforced upstream in the graph.

    Args:
        action: The remediation action to take, e.g. "rolling_restart" or
            "scale_replicas".
        target: The Kubernetes resource the action applies to, e.g.
            "payment-service-deployment".

    Returns:
        A human-readable confirmation string describing what was executed.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    return (
        f"[{timestamp}] Executed action='{action}' on target='{target}'. "
        f"Kubernetes API acknowledged the request; rollout status: "
        f"in-progress. This was a simulated execution for demo purposes."
    )

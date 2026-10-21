"""Build the LoRA fine-tuning dataset for the incident-triage RCA model.

Generates train/valid JSONL files in MLX chat format. Each example mirrors
the exact prompt shape used by `rca_synthesizer_node` in graph.py, with an
assistant response that is strict, schema-conformant JSON. The goal is to
teach the base model to consistently emit short, machine-usable
`remediation_action` slugs (e.g. "rolling_restart") instead of prose
sentences, which is the quality gap observed from the un-tuned model.
"""

from __future__ import annotations

import json
import os
import random

PROMPT_TEMPLATE = (
    "You are an SRE copilot performing root cause analysis for a "
    "production incident. Using the telemetry findings and runbook "
    "guidance below, produce a concise RCA and propose exactly one "
    "remediation action.\n\n"
    "Service: {service}\n\n"
    "Telemetry findings:\n{telemetry}\n\n"
    "Runbook guidance:\n{runbook}\n"
)

RESPONSE_SCHEMA_NOTE = (
    "Respond with ONLY a JSON object with exactly these keys: "
    '"rca_summary" (1-2 sentences), "remediation_action" (a short '
    'snake_case slug, e.g. "rolling_restart"), "remediation_target" '
    '(the k8s resource name), "remediation_reason" (one sentence).'
)

# Each scenario: service, telemetry blurb, runbook blurb, and the target
# structured output. Action slugs are drawn from a small fixed vocabulary
# so the model learns a consistent, constrained action space.
SCENARIOS = [
    dict(
        service="payment-service",
        telemetry=(
            "--- Logs ---\nERROR PostgresConnectionPoolTimeout: timeout after "
            "30000ms waiting for connection from pool (size=50, overflow=10, "
            "checked_out=60)\nWARN slow query detected duration=28410ms "
            "(idle_in_transaction=true)\n--- Metrics ---\n"
            '{"http_5xx_rate_pct": 18.7, "latency_p99_ms": 6240, '
            '"connection_pool_utilization_pct": 99.4}'
        ),
        runbook=(
            "payment-service connection pool exhaustion: caused by leaked "
            "connections or idle-in-transaction queries. Mitigate with a "
            "rolling restart of the deployment to release stuck connections; "
            "follow up by fixing the leaking code path."
        ),
        rca_summary=(
            "Postgres connection pool exhaustion on payment-service, driven "
            "by idle-in-transaction queries holding connections open; pool "
            "utilization at 99.4% is causing 504s and elevated p99 latency."
        ),
        action="rolling_restart",
        target="payment-service-deployment",
        reason=(
            "A rolling restart releases the stuck/leaked connections and "
            "restores pool headroom immediately."
        ),
    ),
    dict(
        service="auth-service",
        telemetry=(
            "--- Logs ---\nERROR TokenValidationTimeout: redis client timed "
            "out validating token after 2000ms\nWARN cache eviction observed: "
            "evicted_keys=184 policy=allkeys-lru used_memory=0.98*maxmemory\n"
            "--- Metrics ---\n"
            '{"http_5xx_rate_pct": 6.3, "cache_hit_rate_pct": 54.2, '
            '"redis_used_memory_pct": 97.8}'
        ),
        runbook=(
            "auth-service token cache pressure: redis nearing maxmemory "
            "triggers evictions mid-validation. Mitigate by scaling redis "
            "replicas or increasing memory; short-term, scale auth-service "
            "replicas to spread cache load."
        ),
        rca_summary=(
            "Redis session-token cache is at 97.8% memory, evicting keys "
            "mid-validation and causing TokenValidationTimeout errors; cache "
            "hit rate has dropped to 54.2%."
        ),
        action="scale_replicas",
        target="auth-service-deployment",
        reason=(
            "Scaling auth-service replicas spreads cache load and reduces "
            "contention while the redis memory limit is addressed."
        ),
    ),
    dict(
        service="checkout-service",
        telemetry=(
            "--- Logs ---\nERROR OutOfMemoryError: Java heap space\nWARN GC "
            "pause duration=4200ms (old_gen_usage=96%)\n--- Metrics ---\n"
            '{"memory_utilization_pct": 97.1, "restart_count_1h": 4, '
            '"latency_p99_ms": 8900}'
        ),
        runbook=(
            "checkout-service memory leak pattern: gradual heap growth over "
            "hours ending in OOM kills. Immediate mitigation is a rolling "
            "restart to reclaim memory; root cause requires heap dump "
            "analysis."
        ),
        rca_summary=(
            "checkout-service is OOMing due to old-gen heap exhaustion, "
            "consistent with a slow memory leak; 4 restarts in the last hour "
            "indicate the pod is crash-looping under memory pressure."
        ),
        action="rolling_restart",
        target="checkout-service-deployment",
        reason=(
            "A rolling restart reclaims leaked heap memory immediately and "
            "stops the crash loop while the leak is investigated."
        ),
    ),
    dict(
        service="search-service",
        telemetry=(
            "--- Logs ---\nERROR DiskFullError: no space left on device "
            "(/var/lib/elasticsearch, 2% free)\nWARN index write rejected: "
            "disk watermark exceeded\n--- Metrics ---\n"
            '{"disk_utilization_pct": 98.4, "indexing_latency_ms": 15200, '
            '"http_5xx_rate_pct": 22.1}'
        ),
        runbook=(
            "search-service disk pressure: index segments and old snapshots "
            "fill the volume over time. Mitigate by expanding the persistent "
            "volume claim; restarts will not help since disk is full."
        ),
        rca_summary=(
            "search-service's data volume is at 98.4% utilization, causing "
            "write rejections and indexing failures due to the disk "
            "watermark being exceeded."
        ),
        action="scale_storage",
        target="search-service-pvc",
        reason=(
            "Expanding the persistent volume claim directly addresses the "
            "disk-full condition; a restart would not free space."
        ),
    ),
    dict(
        service="notifications-service",
        telemetry=(
            "--- Logs ---\nERROR SQSConsumerStalled: no messages consumed in "
            "last 10m despite queue depth=48213\nWARN consumer lag "
            "increasing: lag=48213 messages\n--- Metrics ---\n"
            '{"queue_depth": 48213, "consumer_lag_sec": 612, '
            '"cpu_utilization_pct": 12.4}'
        ),
        runbook=(
            "notifications-service queue backlog: low CPU with a growing "
            "queue usually indicates a stuck/deadlocked consumer rather than "
            "a capacity issue. Mitigate with a rolling restart of the "
            "consumer pods."
        ),
        rca_summary=(
            "notifications-service consumers have stalled (CPU at only "
            "12.4% while queue depth grows to 48k), indicating a deadlocked "
            "or hung consumer process rather than insufficient capacity."
        ),
        action="rolling_restart",
        target="notifications-service-deployment",
        reason=(
            "Low CPU with a growing queue points to a stuck consumer, which "
            "a rolling restart will clear without needing more capacity."
        ),
    ),
    dict(
        service="recommendation-service",
        telemetry=(
            "--- Logs ---\nWARN request queue depth exceeds capacity: "
            "pending=890, max_concurrency=100\nERROR 503 Service "
            "Unavailable: thread pool exhausted\n--- Metrics ---\n"
            '{"http_5xx_rate_pct": 31.5, "requests_per_sec": 2400, '
            '"cpu_utilization_pct": 94.8, "replica_count": 3}'
        ),
        runbook=(
            "recommendation-service traffic spike pattern: sustained high "
            "CPU and request queueing under normal error budgets usually "
            "means under-provisioned replicas for current load. Mitigate by "
            "scaling out replicas."
        ),
        rca_summary=(
            "recommendation-service is under-provisioned for current "
            "traffic: CPU at 94.8% across only 3 replicas with 2400 rps is "
            "exhausting the thread pool and causing 503s."
        ),
        action="scale_replicas",
        target="recommendation-service-deployment",
        reason=(
            "Scaling out replicas adds capacity to absorb the current "
            "traffic level and relieve CPU and thread-pool saturation."
        ),
    ),
    dict(
        service="billing-service",
        telemetry=(
            "--- Logs ---\nERROR DatabaseConnectionRefused: primary db host "
            "unreachable (billing-db-primary:5432)\nERROR failover not "
            "triggered: replica lag=0s, health check still passing on "
            "replica\n--- Metrics ---\n"
            '{"http_5xx_rate_pct": 64.2, "db_primary_up": 0, "db_replica_up": 1}'
        ),
        runbook=(
            "billing-service primary database outage: when the primary is "
            "unreachable but a healthy replica exists, promote the replica "
            "to primary (failover) rather than restarting the service."
        ),
        rca_summary=(
            "billing-service's primary database is unreachable while a "
            "healthy replica exists with zero lag, causing widespread 5xx "
            "errors on all database-dependent requests."
        ),
        action="failover_database",
        target="billing-db-replica",
        reason=(
            "Promoting the healthy, zero-lag replica restores database "
            "availability immediately without data loss."
        ),
    ),
    dict(
        service="image-processing-service",
        telemetry=(
            "--- Logs ---\nWARN cache miss rate elevated: 91% misses on "
            "thumbnail cache\nERROR CDN origin overload: origin_requests_per_sec=4200 "
            "(expected <500)\n--- Metrics ---\n"
            '{"cache_hit_rate_pct": 9.0, "origin_load_pct": 98.0, '
            '"latency_p99_ms": 7400}'
        ),
        runbook=(
            "image-processing-service cache poisoning/invalidation pattern: "
            "a bad deploy or TTL misconfiguration can drop the cache hit "
            "rate, overloading the origin. Mitigate by clearing and "
            "rewarming the cache."
        ),
        rca_summary=(
            "image-processing-service's thumbnail cache hit rate has "
            "collapsed to 9%, pushing nearly all traffic to the origin "
            "(98% load) and driving up latency."
        ),
        action="clear_cache",
        target="image-processing-thumbnail-cache",
        reason=(
            "Clearing and rewarming the cache resolves the stale/poisoned "
            "entries causing the abnormal miss rate."
        ),
    ),
]


def _build_example(s: dict) -> dict:
    prompt = PROMPT_TEMPLATE.format(
        service=s["service"], telemetry=s["telemetry"], runbook=s["runbook"]
    ) + "\n" + RESPONSE_SCHEMA_NOTE
    response = json.dumps(
        {
            "rca_summary": s["rca_summary"],
            "remediation_action": s["action"],
            "remediation_target": s["target"],
            "remediation_reason": s["reason"],
        }
    )
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


def main() -> None:
    random.seed(7)
    examples = [_build_example(s) for s in SCENARIOS]
    random.shuffle(examples)

    n_valid = max(1, len(examples) // 5)
    valid, train = examples[:n_valid], examples[n_valid:]

    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "train.jsonl"), "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")

    with open(os.path.join(out_dir, "valid.jsonl"), "w") as f:
        for ex in valid:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(train)} train examples, {len(valid)} valid examples to {out_dir}")


if __name__ == "__main__":
    main()

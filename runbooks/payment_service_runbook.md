# Runbook: payment-service

## Service Overview
`payment-service` handles checkout, authorization, and settlement calls between
the storefront and the card-processing network. It is backed by a PostgreSQL
primary with a fixed-size connection pool (max 50 connections per pod).

## Known Failure Mode: HTTP 504 Gateway Timeout

### Symptoms
- Upstream load balancer / API gateway returns `504 Gateway Timeout` on
  `/v1/payments/authorize` and `/v1/payments/capture`.
- `payment-service` p99 latency climbs above 5000ms while p50 stays near
  baseline (bimodal latency distribution — a sign of queued requests waiting
  on a resource, not uniformly slow processing).
- Error rate on 5xx responses rises sharply, often in a step function rather
  than a gradual slope.
- Correlates with deploy events, traffic spikes (e.g. flash sales), or a
  long-running downstream query holding connections open.

### Likely Root Cause: Postgres Connection Pool Exhaustion

Symptoms that confirm this root cause:
- `connection_pool_utilization` metric at or near 100%.
- Application logs show `PostgresConnectionPoolTimeout` or
  `TimeoutError: QueuePool limit of size 50 overflow 10 reached` style stack
  traces.
- Number of idle-in-transaction connections on the Postgres side is elevated.
- No corresponding spike in CPU/memory on the database host — this is a
  connection accounting problem, not raw resource starvation.

### Recommended Remediation

1. **Immediate mitigation:** Perform a **rolling restart of
   `payment-service-deployment`** to clear leaked/stuck connections and
   release pool slots held by stuck goroutines/threads.
   ```
   kubectl rollout restart deployment/payment-service-deployment -n payments
   ```
2. **Verify recovery:** Confirm `connection_pool_utilization` drops below 60%
   and p99 latency returns under 500ms within 5 minutes of the restart
   completing.
3. **Follow-up (non-urgent):** File a ticket to audit for connections that
   are not being released on query timeout/error paths, and consider lowering
   the statement timeout or increasing pool size if restarts recur within 24
   hours.

### Escalation
If a rolling restart does not resolve the 504s within 10 minutes, escalate to
the Payments on-call engineer — this may indicate a downstream database
incident rather than an application-level leak.

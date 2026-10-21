# Runbook: auth-service

## Service Overview
`auth-service` issues and validates session tokens for all downstream APIs.
It uses a Redis cluster as a hot cache for token validation and falls back to
the primary token store on cache miss.

## Known Failure Mode: Redis Cache Eviction Failures

### Symptoms
- Spike in `token_validation_latency_ms`, especially p95/p99, while cache hit
  rate drops noticeably.
- Redis logs / metrics show evictions under the `allkeys-lru` policy
  happening far more frequently than normal, often because memory usage is
  pinned near `maxmemory`.
- Increased load on the fallback token store as validations miss cache and
  fall through, which can cause a secondary latency cascade.
- Intermittent `EvictionPolicyError` or stale-token mismatches in
  `auth-service` logs when a token is evicted mid-validation.

### Likely Root Cause: Token Validation Timeouts from Stale/Evicted Cache

Symptoms that confirm this root cause:
- `cache_hit_rate` metric trending down over minutes to hours.
- `fetch_service_logs` shows repeated `TokenValidationTimeout` traces
  originating from the Redis client, not the primary datastore.
- Replica count for `auth-service` has not scaled with increased traffic,
  so each pod is doing more validation work against a cache that is already
  under memory pressure.

### Recommended Remediation

1. **Immediate mitigation:** Flush the stale token cache keys in Redis to
   force a clean rebuild of hot entries:
   ```
   redis-cli --cluster call <redis-host>:6379 FLUSHDB ASYNC
   ```
   (Scope the flush to the token namespace in production; a full `FLUSHDB`
   is acceptable in this simulated environment.)
2. **Scale out:** Increase `auth-service` replicas from **2 to 4** to
   spread validation load and reduce per-pod cache pressure:
   ```
   kubectl scale deployment/auth-service --replicas=4 -n identity
   ```
3. **Verify recovery:** Confirm `cache_hit_rate` returns above 90% and
   `token_validation_latency_ms` p99 drops under 200ms within 5 minutes.

### Escalation
If cache hit rate does not recover after the flush and scale-out, escalate to
the Identity on-call engineer to check for a misconfigured `maxmemory-policy`
or a Redis node failure within the cluster.

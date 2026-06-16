# Semantic Caching Layer for LLM APIs — Build Plan

## 1. What you're building
A drop-in middleware proxy between any app and an LLM provider. It embeds each
prompt, looks up semantically similar past requests in a vector store, and serves
cached responses on a hit — cutting cost and latency. Mirrors the OpenAI API so
adoption = changing one base URL.

## 2. Locked decisions
| Decision | Choice |
|---|---|
| Repo | `semantic-llm-cache` (separate) |
| Language / proxy | Python 3.11+ / FastAPI |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store + cache | Redis Stack (RedisVL) — one container does vectors + TTL |
| Dev/test chat provider | Ollama (local, free) |
| Hosted chat provider | OpenAI |
| Third provider (Phase 2+) | Anthropic (adapter, demonstrates routing) |
| Monitoring | Prometheus + Grafana |
| Packaging | Docker + docker-compose |

## 3. The two things that make this stand out
1. **Cache precision** — % of cache hits that were correct to serve, measured
   against a labeled should-hit/should-miss eval set. Reporting precision (not just
   hit rate) is the senior move.
2. **A documented workload mix** — publish the mix (e.g. 40% repeats / 30%
   paraphrases / 30% unique) and report savings against it.

Headline target:
> "Drop-in caching layer that cut LLM API cost by X% and P95 latency by Y% at Z%
> cache precision on a 2,000-request load test (40/30/30 repeat/paraphrase/unique)."

## 4. Phases
- **Phase 0 — Setup.** uv project, Docker stack (api + redis + prometheus +
  grafana), config, health/readiness. ✅
- **Phase 1 — Cache index & similarity engine.** Embed prompt, RedisVL HNSW/cosine
  search, namespace isolation (model + system + params), hit rule (cosine ≥ 0.95),
  single-embed query/store, overhead benchmark. ✅
- **Phase 2 — Drop-in proxy API.** `/v1/chat/completions` mirroring OpenAI;
  provider routing by `model` (Ollama/OpenAI/Anthropic); streaming with
  buffer-to-store on miss.
- **Phase 3 — Cache policies & eviction.** TTL tiers; threshold tuner endpoint;
  adaptive thresholds by request type; no-cache path; single-flight lock.
- **Phase 4 — Monitoring & analytics.** Prometheus metrics (hit rate, cached vs
  uncached P50/P95/P99, cost saved, similarity histogram); Grafana dashboard;
  near-miss analyzer.
- **Phase 5 — Containerize & load test.** 2,000+ request workload (40/30/30) on
  Ollama; capture hit-rate convergence, latency percentiles, cost savings, cache
  precision.
- **Phase 6 — Polish for portfolio.** Demo flow; README headline numbers;
  architecture diagram.

## 5. Risks
| Risk | Severity | Mitigation |
|---|---|---|
| False cache hits (wrong answers) | HIGH | precision eval set; adaptive thresholds |
| Savings look cherry-picked | HIGH | publish workload mix; report against it |
| Embed+search overhead > savings on misses | MED | benchmark in Phase 1; keep ≪ provider latency |
| Streaming reassembly edge cases | MED | buffer-then-store; test partial streams |
| Key/prompt leakage (it's a proxy) | MED | env secrets; no prompt logging by default |
| Cache stampede on cold popular query | LOW | single-flight lock per key |

## 6. Complexity
MEDIUM — ~12–14 focused days. No GPU, no dataset, no external blockers.

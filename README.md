# Semantic LLM Cache

A drop-in middleware proxy that sits between your app and any LLM provider,
detects semantically similar requests, and serves cached responses instantly —
cutting API cost and latency.

It mirrors the OpenAI API, so adopting it means changing one base URL.

## Status

| Phase | Scope | State |
|-------|-------|-------|
| 0 | Repo scaffold, Docker stack, health endpoints | ✅ done |
| 1 | Cache index + similarity engine | ✅ done |
| 2 | Drop-in proxy API (OpenAI contract, streaming, routing) | ✅ done |
| 3 | Cache policies & eviction | ⬜ next |
| 4 | Monitoring & analytics | ⬜ |
| 5 | Containerize & load test | ⬜ |
| 6 | Portfolio polish | ⬜ |

See [PLAN.md](PLAN.md) for the full build plan.

## Stack

Python 3.11+ · FastAPI · OpenAI `text-embedding-3-small` · Redis Stack (RedisVL)
· Prometheus + Grafana · Docker Compose. Dev/test chat provider: **Ollama** (free, local).

## Proxy usage

Point any OpenAI client at the proxy by changing the base URL:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed-for-ollama")
client.chat.completions.create(
    model="llama3.2:3b",                     # → Ollama (local). gpt-* → OpenAI, claude-* → Anthropic
    messages=[{"role": "user", "content": "hello"}],
)
```

The response carries an `X-Cache: HIT|MISS` header. Streaming (`stream=True`) is
supported on every path — cache hits replay instantly, misses stream live while
buffering the full response to store.

## Setup

```bash
# 1. Configure secrets (never commit .env)
cp .env.example .env
#    then edit .env and add your OWN OpenAI / Anthropic keys

# 2. Install dependencies
uv sync

# 3. Start the infra (Redis Stack + Prometheus + Grafana)
docker-compose up -d redis prometheus grafana

# 4. Run the API locally
uv run uvicorn app.main:app --reload
#    GET http://localhost:8000/health        -> {"status":"ok"}
#    GET http://localhost:8000/health/ready   -> {"status":"ready"}  (needs Redis)
```

## Tests

```bash
docker-compose up -d redis      # integration tests need Redis Stack
uv run pytest                   # unit tests run without it; integration tests skip if absent
```

## Phase 1 overhead benchmark

```bash
uv run python scripts/bench_phase1.py   # embed + search latency (needs OPENAI_API_KEY)
```

## Security

- Secrets live only in `.env` (git-ignored). Never commit keys, never paste them anywhere.
- The proxy never logs prompt contents by default.

## Ports

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| RedisInsight | http://localhost:8001 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (anon enabled; admin/admin) |

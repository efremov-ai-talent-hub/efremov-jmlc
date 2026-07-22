# efremov-jmlc — mini showcase

## Services

| Service            | What it is                                   | Host port (default) |
| ------------------ | -------------------------------------------- | ------------------- |
| `minio`            | S3-compatible object store                   | API `39010`, console `39011` |
| `minio-init`       | one-shot: creates the `warehouse` bucket     | —                   |
| `iceberg-postgres` | JDBC catalog DB for the Iceberg REST service | —                   |
| `iceberg-rest`     | Iceberg REST catalog (→ MinIO)               | `38181`             |
| `iceberg-init`     | one-shot: creates the `raw` namespace        | —                   |
| `trino`            | query engine — **only** the `iceberg` catalog | `38080`            |
| `prefect`          | Prefect 3 server (UI + API), no deployments  | `34200`             |
| `prefect-worker`   | polls the work pool; nothing deployed yet    | —                   |
| `prefect-postgres` | Prefect's backing DB                         | —                   |
| `litellm`          | patched LiteLLM proxy (LLM + ASR routing)    | `34000`             |
| `litellm-db`       | LiteLLM's backing DB                         | —                   |
| `gigaam`           | self-hosted GigaAM ASR (Russian speech-to-text) | `38103`          |
| `grafana`          | Grafana (no datasources/metrics wired yet)   | `33000`             |
| `mcp-grafana`      | Grafana MCP server (proxied via LiteLLM `/mcp`) | —                |

Data flow: **MinIO (S3) → Iceberg REST (+ Postgres catalog) → Trino**.
LiteLLM exposes two model groups — `gigachat-lite` (chat) and
`transcription-gigaam` (speech-to-text, routed to the local `gigaam` service) —
and re-exposes the Grafana MCP server under its own `/mcp`. Grafana currently has
no datasources wired up.

## Quick start

```bash
cp .env.example .env      # or: make env
make up                   # build images + start everything (detached)
make status               # watch containers come up
```

Then:

- MinIO console — http://localhost:39011  (user `minio` / pass `minio123`)
- Iceberg REST config — http://localhost:38181/v1/config
- Trino UI — http://localhost:38080  (user: any, e.g. `dbt`)
- Prefect UI — http://localhost:34200  (credentials from `PREFECT_SERVER_API_AUTH_STRING`)
- Grafana — http://localhost:33000  (user `admin` / pass `admin`)
- LiteLLM — http://localhost:34000/health (set `LITELLM_GIGACHAT_CREDENTIALS`
  in `.env` for `gigachat-lite` to actually route completions)

Smoke-test Trino → Iceberg once everything is healthy:

```bash
docker compose exec trino trino --execute "SHOW SCHEMAS FROM iceberg;"
```

Transcribe a file through the proxy (GigaAM runs on CPU — expect it to be slower
than realtime, and the first call after a fresh start waits on the model load):

```bash
curl -s http://localhost:34000/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-master-dev-change-me" \
  -F model=transcription-gigaam \
  -F file=@call.mp3
```

Tear down:

```bash
make down       # keep volumes
make clean      # also delete volumes (destroys data)
```

## Notes

- **LiteLLM is a custom image**: upstream LiteLLM + three build-time patches
  (GigaChat `finish_reason`, A2A method map, OTEL non-dict guard). See
  `infra/litellm/Dockerfile` and `infra/litellm/patches/`.
- **Trino config** lives in `infra/trino/etc/`. The iceberg catalog reads MinIO
  creds and the warehouse bucket from the environment via `${ENV:VAR}`, passed in
  from `.env` by the `trino` service — so they never skew from the rest of the stack.
- **GigaAM** (`ai/infra/gigaam/`) is a thin OpenAI-compatible wrapper around the
  GigaAM ASR model, built locally and reached only through LiteLLM. It is the
  heaviest service here: CPU-only inference, weights downloaded into the
  `gigaam-cache` volume on first boot, model kept warm in RAM. Set
  `GIGAAM_HF_TOKEN` in `.env` for longform audio — the VAD weights it needs are a
  gated HuggingFace repo, and without an authorised token only short files work.
  `ai/infra/gigaam/trials/` is the standalone experiment stand that picked this
  model; it has its own compose file and is not part of this stack.
- **Prefect** runs a server plus a worker on the pool named by `PREFECT_WORK_POOL`,
  created on first worker start. Nothing is deployed to it yet, so the worker just
  polls an empty pool; it uses the stock Prefect image and will need a build of its
  own once flows arrive with their dependencies. The UI and API require the
  credentials in `PREFECT_SERVER_API_AUTH_STRING` — which must not be blank: an
  empty value switches the server's auth on while leaving the client unable to
  authenticate, and `GET /api/health` stays exempt, so the container still looks
  healthy while every real call gets 401.
- **Apple Silicon**: the `prefect` server exits with SIGILL on linux/arm64, so the
  stack does not come up on an arm64 dev machine. The deploy target is amd64 and is
  unaffected.
- Host ports bind to `127.0.0.1` only.

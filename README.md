# efremov-jmlc — mini showcase

## Services

| Service            | What it is                                   | Host port (default) |
| ------------------ | -------------------------------------------- | ------------------- |
| `minio`            | S3-compatible object store                   | API `39010`, console `39011` |
| `minio-init`       | one-shot: creates the warehouse / raw-files / eval-datasets buckets | — |
| `iceberg-postgres` | JDBC catalog DB for the Iceberg REST service | —                   |
| `iceberg-rest`     | Iceberg REST catalog (→ MinIO)               | `38181`             |
| `iceberg-migrate`  | one-shot: applies `migrations/*.sql` via Trino | —                 |
| `trino`            | query engine — **only** the `iceberg` catalog | `38080`            |
| `prefect`          | Prefect 3 server (UI + API), no deployments  | `34200`             |
| `prefect-worker`   | runs the flows; built with the pipeline code | —                   |
| `prefect-postgres` | Prefect's backing DB                         | —                   |
| `litellm`          | patched LiteLLM proxy (LLM + ASR routing)    | `34000`             |
| `litellm-db`       | LiteLLM's backing DB                         | —                   |
| `gigaam`           | self-hosted GigaAM ASR (Russian speech-to-text) | `38103`          |
| `prometheus`       | scrapes gigaam and litellm metrics           | —                   |
| `grafana`          | Grafana + provisioned Prometheus datasource  | `33000`             |
| `mcp-grafana`      | Grafana MCP server (proxied via LiteLLM `/mcp`) | —                |

Data flow: **MinIO (S3) → Iceberg REST (+ Postgres catalog) → Trino**.
LiteLLM exposes five model groups — `gigachat-lite` / `-pro` / `-max`,
`gigachat-lightning-local` (a model served outside the stack, reached over a
tunnel) and `transcription-gigaam` (speech, routed to the local `gigaam` service) —
and re-exposes the Grafana MCP server under its own `/mcp`. Grafana is provisioned
from files with a Prometheus datasource and the GigaAM ASR dashboard.

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

## Verifying it works

Four commands, each exercising a path the others do not. Worth running in this
order after a deploy — a green deploy means the containers started, not that any
of this answers.

**Trino → Iceberg REST → MinIO.** Lists the tables the migrations created:

```bash
docker compose exec -T trino trino --execute "SHOW TABLES FROM iceberg.analysis"
```

Note this reads catalog metadata only and never touches object storage. To prove
writes work — a distinct failure mode, since MinIO answers `507` on writes past
its free-space threshold while still serving reads — create and drop a table:

```bash
docker compose exec -T trino trino --execute "CREATE TABLE iceberg.analysis._probe (x int)"
docker compose exec -T trino trino --execute "DROP TABLE iceberg.analysis._probe"
```

**GigaAM through the proxy.** Proves the wrapper, the model weights and LiteLLM's
audio routing all work together. CPU inference is slower than realtime, and the
first call after a fresh start waits on the model load:

```bash
curl -s http://localhost:34000/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-master-dev-change-me" \
  -F model=transcription-gigaam \
  -F file=@seeds/audio/v1.mp3
```

**LiteLLM's MCP gateway → Grafana.** Proves the whole chain: the proxy forwards
MCP calls, `mcp-grafana` reaches Grafana, and the dashboard provisioning landed.
Returns the `gigaam-asr` dashboard:

```bash
curl -s -X POST http://localhost:34000/mcp/ \
  -H "Authorization: Bearer sk-master-dev-change-me" \
  -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" \
  -H "x-mcp-servers: grafana" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"grafana-search_dashboards","arguments":{"query":"gigaam"}}}'
```

**The same gateway down to real metrics.** A different path from the one above:
dashboards come from Grafana's own database, while this goes on through the
datasource to Prometheus. `gigaam_model_loaded` is the one metric that exists
before any transcription has run:

```bash
curl -s -X POST http://localhost:34000/mcp/ \
  -H "Authorization: Bearer sk-master-dev-change-me" \
  -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" \
  -H "x-mcp-servers: grafana" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"grafana-query_prometheus","arguments":{"datasourceUid":"prometheus","expr":"gigaam_model_loaded","queryType":"instant","startTime":"now-5m","endTime":"now"}}}'
```

## Running the pipeline

Seed calls from local audio, then trigger a flow. Deployments are registered by
the deploy itself; `make deploy-flows` is only for registering them by hand.

```bash
make seed
docker compose exec -T prefect-worker prefect deployment run 'analysis-transcription/transcription'
docker compose exec -T prefect-worker prefect deployment run 'analysis-call-report/call-report-v3'
docker compose exec -T prefect-worker prefect deployment run 'eval-export-dataset/eval-export'
```

Each `deployment run` prints the flow run's UUID. Follow it with that id — ASR runs
on CPU here, so a call takes longer than its own duration to transcribe:

```bash
RUN_ID=f7e57bf6-633b-43f7-8698-afcd3d466fab   # the UUID the trigger printed

docker compose exec -T prefect-worker prefect flow-run inspect "$RUN_ID" | grep state_name
docker compose exec -T prefect-worker prefect flow-run logs "$RUN_ID" | tail -40
```

The flow logs its own progress (`transcription candidates: 1`) because the worker
is configured to surface the `pipelines` and `ai` loggers, not just Prefect's own.

Then read the result. Summary first — a transcript is long enough to be unpleasant
in a terminal:

```bash
docker compose exec -T trino trino --execute \
"SELECT call_id, version, is_current, language, duration_seconds, model, length(text) AS chars
 FROM iceberg.analysis.transcriptions ORDER BY created_at DESC"

docker compose exec -T trino trino --execute \
"SELECT text FROM iceberg.analysis.transcriptions WHERE is_current = true"
```

`analysis.llm_calls` is the journal of what was attempted, failures included — a
failed attempt carries a null `artifact_id` but keeps its `call_id`, so it stays
attributable. Its `request_id` is the same id LiteLLM files the call under, which
is how a result here is traced back to the request the proxy actually sent:

```bash
docker compose exec -T trino trino --execute \
"SELECT call_id, artifact_type, status, model, request_id, latency_ms,
        substr(coalesce(error, ''), 1, 200) AS err
 FROM iceberg.analysis.llm_calls ORDER BY created_at DESC"
```

A flow that finishes `Completed` while `transcriptions` stays empty means it found
no candidates — check that seeding landed, with
`SELECT call_id, audio_key FROM iceberg.analysis.calls`.

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
  created on first worker start. Flows live in `pipelines/flows/` and are
  registered by the deploy itself, from inside the worker — that is where the code
  and its dependencies are. `make deploy-flows` does the same by hand, for when
  you are iterating without deploying. Deployments carry no
  schedule on purpose — every run costs LLM calls, and a stand that spends money
  on a timer is worse than one you trigger. The UI and API require the
  credentials in `PREFECT_SERVER_API_AUTH_STRING` — which must not be blank: an
  empty value switches the server's auth on while leaving the client unable to
  authenticate, and `GET /api/health` stays exempt, so the container still looks
  healthy while every real call gets 401.
- **Choosing a model.** The flows ask the proxy for a *model group*, never a
  provider model id: `ANALYSIS_CHAT_MODEL` for analysis and `ANALYSIS_WHISPER_MODEL`
  for speech, both in `.env`. Deployments deliberately carry no model parameter —
  a value there would be baked in at registration and would win over `.env` — so
  changing the variable and restarting the worker is the whole procedure.
  `gigachat-lite` / `-pro` / `-max` share one credential; `gigachat-lightning-local`
  points at whatever `LOCAL_LLM_BASE_URL` names, for running the analysis against a
  model of your own. That address has to be reachable **from inside the LiteLLM
  container**, which rules out a loopback one — `127.0.0.1` there is the container
  itself. A model served on the deploy host is reached at `host.docker.internal`,
  which the service maps to the host gateway (Docker Desktop provides that name on
  its own; Linux does not).
- **Apple Silicon**: the `prefect` server exits with SIGILL on linux/arm64, so the
  stack does not come up on an arm64 dev machine. The deploy target is amd64 and is
  unaffected.
- **Data model** lives in `migrations/*.sql` and is applied through Trino by the
  one-shot `iceberg-migrate` service on every deploy. Statements are idempotent
  (`CREATE ... IF NOT EXISTS`); the Trino CLI exits non-zero on the first failing
  statement, and `ci-deploy.sh` waits on the container so a failed migration fails
  the deploy. The `DESCRIBE` pass afterwards covers the other case — a table
  dropped or renamed out of band. Adding a column later means a
  new numbered file with `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`; a tracking
  table only becomes necessary once a migration stops being idempotent.
- **Observability** is provisioned from `infra/observability/`, not clicked in:
  Prometheus scrapes `gigaam` and `litellm`, and Grafana gets the datasource plus
  the GigaAM ASR dashboard on every deploy. This is also what makes both visible
  to the Grafana MCP server — it can only reach what Grafana knows about, so a
  hand-made datasource on one host would leave the MCP path untestable on another.
- Host ports bind to `127.0.0.1` only.

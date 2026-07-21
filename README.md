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
| `prefect-postgres` | Prefect's backing DB                         | —                   |
| `litellm`          | patched LiteLLM proxy (only `gigachat-lite`) | `34000`             |
| `litellm-db`       | LiteLLM's backing DB                         | —                   |
| `grafana`          | Grafana (no datasources/metrics wired yet)   | `33000`             |
| `mcp-grafana`      | Grafana MCP server (proxied via LiteLLM `/mcp`) | —                |

Data flow: **MinIO (S3) → Iceberg REST (+ Postgres catalog) → Trino**.
LiteLLM exposes one model group (`gigachat-lite`) and re-exposes the Grafana
MCP server under its own `/mcp`. Grafana currently has no datasources wired up.

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
- Prefect UI — http://localhost:34200
- Grafana — http://localhost:33000  (user `admin` / pass `admin`)
- LiteLLM — http://localhost:34000/health (set `LITELLM_GIGACHAT_CREDENTIALS`
  in `.env` for `gigachat-lite` to actually route completions)

Smoke-test Trino → Iceberg once everything is healthy:

```bash
docker compose exec trino trino --execute "SHOW SCHEMAS FROM iceberg;"
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
- Host ports bind to `127.0.0.1` only.

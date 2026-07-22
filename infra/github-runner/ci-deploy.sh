#!/usr/bin/env bash
set -euo pipefail

STACK_DIR="/home/markefremov/efremov-jmlc"
REF="${1:-origin/main}"

cd "$STACK_DIR"

echo "==> fetch + checkout ${REF}"
git fetch --prune origin
git checkout -f --detach "${REF}"

echo "==> .env"
if [ ! -f .env ]; then
  cp .env.example .env
  echo "created .env from .env.example"
else
  echo "keeping existing .env"
fi

echo "==> pull готовых образов"
docker compose pull --ignore-buildable

echo "==> build локальных образов"
docker compose build --pull

echo "==> up -d"
docker compose up -d --remove-orphans

# `up -d` returns once a one-shot container has *started* and never looks at its
# exit code, so a failed migration would otherwise print "deploy OK". Wait for it
# and surface the logs on failure.
echo "==> migrations"
# `docker compose wait <service>` exits with the container's own exit code, so
# test that directly — its stdout is prose ("container ... exited with status
# code 0"), not a bare number.
#
# Bounded, because `wait` has no timeout of its own and the job's concurrency
# group does not cancel in progress: one hung wait blocks every later deploy
# until someone notices. 600s is far above the seconds this normally takes, so
# hitting it means something is wrong rather than slow.
# `|| rc=$?` rather than `if ! …`: inside an `if !` branch $? is the status of
# the negation (always 0), so the real exit code would be lost. The `||` also
# keeps `set -e` from killing the script before we can report.
rc=0
timeout -k 30 600 docker compose wait iceberg-migrate || rc=$?
if [ "$rc" -ne 0 ]; then
  case "$rc" in
    124 | 137) echo "!! migrations timed out after 600s" ;;
    *) echo "!! migrations failed (exit $rc)" ;;
  esac
  # Bounded: the likeliest cause of a 600s wait is a wedged docker daemon, and an
  # unbounded `logs` here would re-block the concurrency group one line after the
  # code added to prevent it. `|| true` so this line's status does not become the
  # script's under pipefail.
  timeout 60 docker compose logs --no-log-prefix iceberg-migrate 2>&1 | tail -40 || true
  # `wait` only observes; killing it leaves the container running. Without this,
  # the next deploy's `up -d` is a no-op for an already-running container (the
  # migrations are bind-mounted, so editing SQL does not change its config hash),
  # and it would wait on the same wedged container again — deploys would stop
  # hanging forever but could never succeed.
  timeout 60 docker compose rm -sf iceberg-migrate >/dev/null 2>&1 || true
  exit 1
fi
echo "migrations ok"

# Registering deployments belongs to the deploy, not to a human afterwards:
# without it a host comes up carrying the flow code with nothing runnable
# against it — green deploy, unusable stand. `prefect deploy` is idempotent, so
# re-running it every time is the point.
#
# Bounded and non-zero-fatal for the same reasons as the migration wait above.
echo "==> deployments"
rc=0
timeout -k 30 300 docker compose exec -T prefect-worker prefect deploy --all || rc=$?
if [ "$rc" -ne 0 ]; then
  case "$rc" in
    124 | 137) echo "!! deployment registration timed out after 300s" ;;
    *) echo "!! deployment registration failed (exit $rc)" ;;
  esac
  timeout 60 docker compose logs --no-log-prefix prefect-worker 2>&1 | tail -30 || true
  exit 1
fi
echo "deployments ok"

echo "==> deploy OK (${REF})"

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
if ! docker compose wait iceberg-migrate; then
  echo "!! migrations failed"
  docker compose logs --no-log-prefix iceberg-migrate | tail -40
  exit 1
fi
echo "migrations ok"

echo "==> deploy OK (${REF})"

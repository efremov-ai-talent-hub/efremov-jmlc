#!/usr/bin/env bash
set -euo pipefail

STACK_DIR="/home/markefremov/efremov-jmlc"
REF="${1:-origin/main}"

cd "$STACK_DIR"

echo "==> fetch + checkout ${REF}"
git fetch --prune origin
git checkout -f --detach "${REF}"

echo "==> pull готовых образов"
docker compose pull --ignore-buildable

echo "==> build локальных образов"
docker compose build --pull

echo "==> up -d"
docker compose up -d --remove-orphans

echo "==> deploy OK (${REF})"

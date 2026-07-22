SHELL := /bin/bash
COMPOSE := docker compose

.PHONY: help env up down stop status logs build ps clean deploy-flows

help:
	@echo "Targets:"
	@echo "  make env      - create .env from .env.example (if missing)"
	@echo "  make up       - build + start the whole stack (detached)"
	@echo "  make down     - stop and remove containers"
	@echo "  make stop     - stop containers (keep them)"
	@echo "  make status   - show container status"
	@echo "  make logs     - tail logs (all services)"
	@echo "  make clean    - down + remove named volumes (DESTROYS DATA)"
	@echo "  make deploy-flows - register Prefect deployments (runs in the worker)"

env:
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example")

up: env
	$(COMPOSE) up -d --build

build: env
	$(COMPOSE) build

down:
	$(COMPOSE) down

stop:
	$(COMPOSE) stop

status ps:
	$(COMPOSE) ps

logs:
	$(COMPOSE) logs -f --tail=100

clean:
	$(COMPOSE) down -v
# Registered from inside the worker: that is where the flow code and its
# dependencies are installed.
deploy-flows:
	$(COMPOSE) exec -T prefect-worker prefect deploy --all

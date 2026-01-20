.PHONY: build up down logs

build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f --tail=200

up-local:
	docker compose -f docker-compose.local.yml up -d

down-local:
	docker compose -f docker-compose.local.yml down

logs-local:
	docker compose -f docker-compose.local.yml logs -f --tail=200

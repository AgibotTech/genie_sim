.PHONY: sync check test lint format typecheck build

sync:
	uv sync --locked --extra pi05

check: lint typecheck test

lint:
	uv run --locked ruff check src tests
	uv run --locked ruff format --check src tests

format:
	uv run --locked ruff check --fix src tests
	uv run --locked ruff format src tests

typecheck:
	uv run --locked mypy

test:
	uv run --locked --extra pi05 pytest

build:
	uv build

.PHONY: check run

check:
	uv run ruff check .
	uv run mypy --strict src tests scripts
	uv run pytest

run:
	uv run python scripts/run_backtest.py

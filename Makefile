.PHONY: env install test test-fast perf lint shots

env:
	conda env create -f environment.yml

install:
	pip install -e .

test:
	pytest

test-fast:
	pytest -m "not ui and not perf"

perf:
	pytest -m perf

lint:
	ruff check .

shots:
	python scripts/make_screenshots.py

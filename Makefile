.PHONY: install test lint typecheck cov build clean publish-test

install:
	pip install -e '.[dev]'

test:
	pytest -q

lint:
	ruff check src tests
	ruff format --check src tests

typecheck:
	mypy src

cov:
	pytest -q --cov=opscopilot --cov-report=term-missing

build: clean
	python -m build
	twine check dist/*

publish-test: build
	twine upload --repository testpypi dist/*

clean:
	rm -rf dist build *.egg-info .pytest_cache .mypy_cache .ruff_cache .coverage

.PHONY: install lint typecheck test check integration

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check .

typecheck:
	mypy src/vocabcraft

test:
	pytest tests/unit

check: lint typecheck test

integration:
	RUN_MT5_INTEGRATION=1 pytest tests/integration


ruff:
	ruff check . --output-format pylint

mypy:
	mypy src/ tests/

test:
	pytest tests/

check: ruff mypy test

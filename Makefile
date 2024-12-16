ruff:
	ruff check . --output-format pylint

mypy:
	mypy src/ tests/

test:
	pytest tests/

check: ruff mypy test

pylint:
	pylint --ignore=$(PYLINT_IGNORED) src/ tests/ --load-plugins=perflint

check: pylint ruff mypy test

python_kernel:
	python -m ipykernel install --user --name=alise_kernel

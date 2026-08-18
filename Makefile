# GNU Makefile that documents and automates common development operations
#              using the GNU make tool (version >= 3.81)
# Development is typically conducted on Linux or Max OS X (with the Xcode
#              command-line tools installed), so this Makefile is designed
#              to work in that environment (and not on Windows).
# USAGE: OG-Core$ make [TARGET]

.PHONY: help
help:
	@echo "USAGE: make [TARGET]"
	@echo "TARGETS:"
	@echo "help       : show help message"
	@echo "clean      : remove .pyc files and local ogcore package"
	@echo "install    : build and install local package"
	@echo "pytest_all : run all tests"
	@echo "pytest_ci  : run same set of tests as GitHub Actions CI"
	@echo "coverage   : generate test coverage report"
	@echo "documentation  : build new Jupyter Book documentation files"
	@echo "format     : format code using ruff and linecheck"

.PHONY: clean
clean:
	@find . -name '*.pyc' -delete
	@find . -maxdepth 1 -name '*cache' -exec rm -r {} +

.PHONY: install
install:
	uv sync --extra dev --extra docs

.PHONY: pytest_all
pytest_all:
	uv run python -m pytest

.PHONY: pytest_ci
pytest_ci:
	uv run python -m pytest -m "not local and not benchmark"

ogcore_JSON_FILES := $(shell ls -l ./ogcore/*json | awk '{print $$9}')

define coverage-cleanup
rm -f .coverage htmlcov/*
endef

COVMARK = "not local and not benchmark"

OS := $(shell uname -s)

.PHONY: coverage
coverage:
	@$(coverage-cleanup)
	uv run python -m pytest -m $(COVMARK) -n 4 --cov=ogcore --cov-report=html --cov-report=term
ifeq ($(OS), Darwin) # on Mac OS X
	@open htmlcov/index.html
else
	@echo "Open htmlcov/index.html in browser to view report"
endif

.PHONY: documentation
documentation:
	uv run --extra docs python -m ipykernel install --user --name=ogcore-dev
	uv run --extra docs jb clean docs
	uv run --extra docs python ./docs/make_params.py
	uv run --extra docs python ./docs/make_vars.py
	uv run --extra docs jb build ./docs/book

.PHONY: format
format:
	uv run ruff format .
	uv run ruff check . --fix
	uv run linecheck . --fix

.PHONY: pip-package
pip-package:
	uv build

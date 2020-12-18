.PHONY: lint test unit-test

SRC := $(shell find synthesized -name '*.py')
VENV_NAME ?= venv
VENV_ACTIVATE = $(VENV_NAME)/bin/activate
PYTHON = $(VENV_NAME)/bin/python3

all: lint unit-test

build: $(SRC)
	$(PYTHON) setup.py bdist_wheel
	./clean_dist.sh
	touch build

test:  venv
	$(PYTHON) -m pytest -v --cov=synthesized  --cov-report=term-missing --junitxml=test-results/junit.xml

fast-test: venv
	$(PYTHON) -m pytest -v -m "fast" --cov=synthesized  --cov-report=term-missing

slow-test:  venv
	$(PYTHON) -m pytest -v -m "slow" --cov=synthesized  --cov-report=term-missing

lint: venv
	$(PYTHON) -m mypy --ignore-missing-import synthesized
	$(PYTHON) -m flake8 --max-line-length=120 synthesized

venv: $(VENV_ACTIVATE)

$(VENV_ACTIVATE): requirements.txt requirements-dev.txt
	test -d $(VENV_NAME) || virtualenv --python=python3 $(VENV_NAME)
	$(PYTHON) -m pip install -U pip==20.3.1
	$(PYTHON) -m pip install -r requirements-dev.txt
	touch $(VENV_ACTIVATE)

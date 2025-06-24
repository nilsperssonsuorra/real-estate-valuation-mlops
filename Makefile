# Makefile for the Automated Housing Valuation MLOps project

# --- Variables ---
PYTHON = python3
DOCKER_IMAGE_NAME = real-estate-valuation-mlops
DOCKER_IMAGE_TAG = latest
# A better practice is to use the git commit hash for the tag:
# DOCKER_IMAGE_TAG = $(shell git rev-parse --short HEAD)

# --- Environment and Dependency Management ---
.PHONY: install requirements
install:
	@echo "--- Setting up the environment from requirements.txt ---"
	$(PYTHON) -m pip install --upgrade pip pip-tools
	pip-sync requirements.txt

requirements:
	@echo "--- Compiling requirements.in to requirements.txt (upgrading packages) ---"
	pip-compile --upgrade requirements.in --output-file=requirements.txt

# --- Code Quality and Testing ---
.PHONY: lint test test-cov all-checks
lint:
	@echo "--- Checking code formatting and style with Ruff ---"
	ruff format . --check
	@echo "--- Linting with Ruff ---"
	ruff check .

test:
	@echo "--- Running tests with Pytest ---"
	$(PYTHON) -m pytest

test-cov:
	@echo "--- Running tests and generating coverage report ---"
	$(PYTHON) -m pytest --cov=src --cov-report=term-missing

all-checks: lint test
	@echo "--- All checks passed successfully! ---"

# --- Local Data and Model Pipeline ---
.PHONY: run-scrape run-clean run-train run-pipeline
run-scrape:
	@echo "--- Running the data scraper (local mode) ---"
	$(PYTHON) src/scrape.py

run-clean:
	@echo "--- Running the data cleaning script (local mode) ---"
	$(PYTHON) src/clean.py

run-train:
	@echo "--- Running the model training script (local mode) ---"
	$(PYTHON) src/train.py

run-pipeline: run-scrape run-clean run-train
	@echo "--- Full local data and model pipeline executed successfully! ---"

# --- Docker Operations ---
.PHONY: docker-build docker-run docker-stop
docker-build:
	@echo "--- Building Docker image: $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG) ---"
	docker build -t $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG) .

docker-run:
	@echo "--- Running Streamlit app inside Docker container ---"
	docker run --rm -p 8501:8501 --name $(DOCKER_IMAGE_NAME) $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)

docker-stop:
	@echo "--- Stopping Docker container: $(DOCKER_IMAGE_NAME) ---"
	docker stop $(DOCKER_IMAGE_NAME)

# --- Cleanup ---
.PHONY: clean
clean:
	@echo "--- Cleaning up Python cache and build artifacts ---"
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -f .coverage
	rm -rf .ruff_cache

# --- Help ---
.PHONY: help
help:
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
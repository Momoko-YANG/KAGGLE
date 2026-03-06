.PHONY: help install clean test lint format train inference docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  install       - Install dependencies and package"
	@echo "  clean         - Remove generated files"
	@echo "  test          - Run tests with coverage"
	@echo "  lint          - Run linting checks"
	@echo "  format        - Format code with black and isort"
	@echo "  train         - Train the model"
	@echo "  inference     - Run inference"
	@echo "  docker-build  - Build Docker image"
	@echo "  docker-run    - Run Docker container"

install:
	pip install --upgrade pip
	pip install -r requirement.txt
	pip install -e .

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	rm -rf build dist .pytest_cache .coverage htmlcov
	rm -rf logs/*.log

test:
	pytest tests/ -v --cov=src --cov-report=term --cov-report=html

test-verbose:
	pytest tests/ -vv --cov=src --cov-report=term-missing

lint:
	flake8 src tests --max-line-length=127
	black --check src tests
	isort --check-only src tests

format:
	black src tests
	isort src tests

train:
	python scripts/train.py

inference:
	python scripts/inference.py

docker-build:
	docker build -t rule-violation-detection:latest .

docker-run:
	docker-compose up

docker-train:
	docker-compose run train

docker-inference:
	docker-compose run inference

setup-dirs:
	mkdir -p data/raw data/processed data/submissions
	mkdir -p models/checkpoints
	mkdir -p logs
	mkdir -p notebooks

pre-commit: format lint test
	@echo "Pre-commit checks passed!"
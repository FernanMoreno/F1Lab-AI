.PHONY: install test lint format typecheck run-dashboard clean help

help:
	@echo "F1Lab-AI - Available commands:"
	@echo "  make install        - Install dependencies"
	@echo "  make test           - Run tests with pytest"
	@echo "  make lint           - Run ruff linter"
	@echo "  make format         - Format code with black"
	@echo "  make typecheck      - Run mypy type checker"
	@echo "  make run-dashboard  - Run Streamlit dashboard"
	@echo "  make clean          - Remove build artifacts"

install:
	@echo "Installing F1Lab-AI..."
	pip install -e ".[data,dashboard,ml,rl,agents,optimization,dev]"

test:
	@echo "Running tests..."
	pytest tests/ -v --cov=reglabsim --cov-report=term-missing

lint:
	@echo "Running ruff linter..."
	ruff check reglabsim agents

format:
	@echo "Formatting code with black..."
	black reglabsim agents tests

typecheck:
	@echo "Running mypy type checker..."
	mypy reglabsim agents --strict

run-dashboard:
	@echo "Starting Streamlit dashboard..."
	streamlit run dashboards/streamlit_app.py

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov/
	@echo "Clean complete."
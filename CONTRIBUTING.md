# Contributing to xMemory-framework

Thank you for your interest in contributing! This document provides guidelines and instructions.

## Code of Conduct

Be respectful, constructive, and inclusive. We're all here to build something useful.

## How to Contribute

### Bug Reports

1. Check existing issues to avoid duplicates
2. Include: OS, PostgreSQL version, pgvector version, error logs, steps to reproduce
3. Use the bug report template

### Feature Requests

1. Describe the use case clearly
2. Explain why existing features don't cover it
3. Include proposed API if possible

### Pull Requests

1. **Fork** the repository
2. **Create a feature branch** (`git checkout -b feature/my-feature`)
3. **Write tests** for your changes
4. **Run the test suite** (`pytest tests/ -v`)
5. **Commit** with clear messages
6. **Open a PR** against the `main` branch

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/xMemory-framework.git
cd xMemory-framework

# Set up development environment
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

# Set up test database
createdb xmemory_test
psql -d xmemory_test -f sql/schema.sql
psql -d xmemory_test -f sql/functions.sql

# Run tests
pytest tests/ -v
```

### Code Style

- Python: Follow PEP 8 (enforced by ruff)
- SQL: Lowercase keywords, snake_case identifiers
- Docs: Markdown with 80-char line width

### Testing

- All new features must include tests
- Maintain >80% code coverage
- Run `pytest tests/ -v --cov=xmemory` before submitting

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

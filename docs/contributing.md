# Contributing

Thank you for your interest in contributing to Black Box Precision Core SDK!

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/blackbox-core-sdk.git
   cd blackbox-core-sdk
   ```
3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Setup

### Running Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=blackboxpcs --cov-report=html
```

### Code Style

We use `black` for formatting and `flake8` for linting:

```bash
black blackboxpcs/
flake8 blackboxpcs/
```

## Contribution Guidelines

### Code Style

- Follow PEP 8 style guide
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Keep functions focused and single-purpose

### Documentation

- Update documentation for any API changes
- Add examples for new features
- Keep docstrings up to date

### Testing

- Write tests for new features
- Ensure all tests pass
- Aim for high test coverage

### Pull Requests

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Write/update tests
4. Update documentation
5. Run tests and linting
6. Submit a pull request

## Areas for Contribution

- Additional explainer implementations
- Performance optimizations
- Documentation improvements
- Example use cases
- Bug fixes

## Questions?

Open an issue on GitHub for questions or discussions.

Thank you for contributing! 🎉


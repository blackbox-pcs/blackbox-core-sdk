# Documentation

This directory contains the source files for the Black Box Precision Core SDK documentation.

## Building the Documentation

### Install Dependencies

```bash
pip install -r requirements-docs.txt
```

### Build Locally

```bash
mkdocs build
```

### Serve Locally

```bash
mkdocs serve
```

Then open http://127.0.0.1:8000 in your browser.

### Deploy

```bash
mkdocs gh-deploy
```

This will deploy the documentation to GitHub Pages.

## Documentation Structure

- `index.md` - Homepage
- `installation.md` - Installation guide
- `getting-started.md` - Quick start tutorial
- `core-concepts.md` - Framework concepts
- `explanation-types.md` - SHAP and LIME details
- `examples.md` - Code examples
- `api/` - API reference documentation
- `use-cases/` - Industry-specific use cases
- `contributing.md` - Contribution guidelines


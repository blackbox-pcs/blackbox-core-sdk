# Installation

Install Black Box Precision Core SDK using your preferred method.

## Requirements

- Python 3.8 or higher
- NumPy >= 1.21.0
- scikit-learn >= 1.0.0
- SHAP >= 0.41.0
- LIME >= 0.2.0.1

## Installation Methods

### Via npm

The package is available on npm:

```bash
npm install blackboxpcs
```

📦 **npm package**: [https://www.npmjs.com/package/blackboxpcs](https://www.npmjs.com/package/blackboxpcs)

After installation via npm, you'll need to install Python dependencies:

```bash
pip install numpy scikit-learn shap lime
```

### Via pip (Python)

#### Install from requirements.txt

```bash
pip install -r requirements.txt
```

#### Install as a package

```bash
pip install -e .
```

This installs the package in editable mode, allowing you to make changes to the source code.

### Install from Source

1. Clone the repository:

```bash
git clone https://github.com/blackbox-pcs/blackbox-core-sdk.git
cd blackbox-core-sdk
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install the package:

```bash
pip install -e .
```

## Optional Dependencies

### Development Dependencies

For development and testing:

```bash
pip install -e ".[dev]"
```

This installs:
- pytest >= 7.0.0
- pytest-cov >= 4.0.0
- black >= 22.0.0
- flake8 >= 5.0.0

### Visualization Dependencies

For visualization capabilities:

```bash
pip install -e ".[visualization]"
```

This installs:
- matplotlib >= 3.5.0
- seaborn >= 0.12.0

## Verify Installation

Test your installation:

```python
import blackboxpcs
print(blackboxpcs.__version__)

from blackboxpcs import BlackBoxPrecision, ExplanationType
print("Installation successful!")
```

## Troubleshooting

### Import Errors

If you encounter import errors:

1. Ensure all dependencies are installed:
   ```bash
   pip install numpy scikit-learn shap lime
   ```

2. Verify Python version:
   ```bash
   python --version  # Should be 3.8+
   ```

### SHAP Installation Issues

If SHAP fails to install:

```bash
pip install --upgrade pip
pip install shap
```

### LIME Installation Issues

If LIME fails to install:

```bash
pip install lime
```

For Windows users, you may need Visual C++ Build Tools.

## Next Steps

- [Quick Start Guide](getting-started.md) - Get started with your first explanation
- [API Reference](api-reference.md) - Explore the complete API
- [Examples](examples.md) - See real-world usage examples


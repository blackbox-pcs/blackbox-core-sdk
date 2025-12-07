# Black Box Precision SDK - Project Structure

```
blackboxpcs/
├── blackboxpcs/              # Main SDK package
│   ├── __init__.py           # Package initialization and exports
│   ├── core.py               # Core BlackBoxPrecision framework
│   ├── explainers.py         # SHAP and LIME explainer implementations
│   └── utils.py              # Utility functions for workflows
│
├── examples/                 # Example usage scripts
│   ├── basic_usage.py       # Basic SDK usage demonstration
│   ├── medical_diagnostics.py # Medical diagnostics case study
│   └── autonomous_systems.py  # Autonomous systems case study
│
├── README.md                  # Comprehensive documentation
├── LICENSE                    # MIT License
├── requirements.txt           # Python dependencies
├── setup.py                   # Package installation script
├── .gitignore                 # Git ignore rules
└── PROJECT_STRUCTURE.md       # This file
```

## Core Components

### `blackboxpcs/core.py`
- `BlackBoxPrecision`: Main framework class
- `ExplanationType`: Enum for SHAP, LIME, or BOTH
- `ExplanationMode`: Enum for GLOBAL (auditing) or LOCAL (operational)

### `blackboxpcs/explainers.py`
- `BaseExplainer`: Base class for all explainers
- `SHAPExplainer`: SHAP implementation for feature attribution
- `LIMEExplainer`: LIME implementation for local explanations

### `blackboxpcs/utils.py`
- `validate_explanation()`: Validate explanation completeness
- `aggregate_explanations()`: Aggregate multiple explanations
- `format_explanation_for_audit()`: Format for audit trails
- `compare_explanations()`: Compare two explanations
- `extract_key_features()`: Extract top contributing features

## Key Features

1. **Dual Explainer Support**: Both SHAP and LIME integrated
2. **Global & Local Modes**: Support for both auditing and operational use
3. **High-Stakes Ready**: Designed for mission-critical applications
4. **Comprehensive Utilities**: Tools for validation, aggregation, and audit trails
5. **Well-Documented**: Extensive examples and documentation

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

See `examples/` directory for complete usage examples:
- Basic usage: `python examples/basic_usage.py`
- Medical diagnostics: `python examples/medical_diagnostics.py`
- Autonomous systems: `python examples/autonomous_systems.py`



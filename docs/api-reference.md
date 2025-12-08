# API Reference

Complete API documentation for Black Box Precision Core SDK.

## Overview

The Black Box Precision SDK provides a unified interface for explainable AI through the following main components:

- **[BlackBoxPrecision](api/blackboxprecision.md)** - Main framework class
- **[SHAPExplainer](api/shapexplainer.md)** - SHAP-based explanations
- **[LIMEExplainer](api/limeexplainer.md)** - LIME-based explanations
- **[Utilities](api/utils.md)** - Helper functions

## Quick Reference

### Core Classes

```python
from blackboxpcs import (
    BlackBoxPrecision,
    SHAPExplainer,
    LIMEExplainer,
    ExplanationType,
    ExplanationMode
)
```

### Utility Functions

```python
from blackboxpcs.utils import (
    validate_explanation,
    aggregate_explanations,
    format_explanation_for_audit,
    compare_explanations,
    extract_key_features
)
```

## Enums

### ExplanationType

```python
class ExplanationType(Enum):
    SHAP = "shap"    # SHAP explanations only
    LIME = "lime"    # LIME explanations only
    BOTH = "both"    # Both SHAP and LIME
```

### ExplanationMode

```python
class ExplanationMode(Enum):
    GLOBAL = "global"  # For auditing and development
    LOCAL = "local"    # For operational oversight
```

## Common Patterns

### Basic Usage

```python
bbp = BlackBoxPrecision(
    model=model,
    explainer_type=ExplanationType.BOTH,
    feature_names=feature_names
)

result = bbp.explain_local(X)
```

### Advanced Usage

```python
# Custom explainer initialization
shap_explainer = SHAPExplainer(
    model=model,
    background_data=X_train[:100],
    algorithm="kernel"
)

result = shap_explainer.explain(X, mode=ExplanationMode.GLOBAL)
```

## Detailed Documentation

- [BlackBoxPrecision](api/blackboxprecision.md) - Main framework API
- [SHAPExplainer](api/shapexplainer.md) - SHAP explainer API
- [LIMEExplainer](api/limeexplainer.md) - LIME explainer API
- [Utilities](api/utils.md) - Utility functions API


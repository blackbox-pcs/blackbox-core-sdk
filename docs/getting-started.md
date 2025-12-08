# Quick Start Guide

Get up and running with Black Box Precision in minutes.

## Basic Usage

### Step 1: Train Your Model

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate or load your data
X, y = make_classification(
    n_samples=200,
    n_features=10,
    n_informative=5,
    random_state=42
)

# Train a black box model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
```

### Step 2: Initialize Black Box Precision

```python
from blackboxpcs import BlackBoxPrecision, ExplanationType

bbp = BlackBoxPrecision(
    model=model,
    explainer_type=ExplanationType.BOTH,  # Use both SHAP and LIME
    feature_names=[f"feature_{i}" for i in range(10)],
    class_names=["class_0", "class_1"]
)
```

### Step 3: Generate Explanations

```python
# Generate local explanation for a single prediction
test_instance = X[0:1]
result = bbp.explain_local(test_instance)

print("Prediction:", result["predictions"])
print("SHAP Explanation:", result["explanations"]["shap"])
print("LIME Explanation:", result["explanations"]["lime"])
```

## Explanation Types

### SHAP Only

```python
bbp = BlackBoxPrecision(
    model=model,
    explainer_type=ExplanationType.SHAP,
    feature_names=feature_names
)

result = bbp.explain_local(X[0:1])
shap_values = result["explanations"]["shap"]["shap_values"]
```

### LIME Only

```python
bbp = BlackBoxPrecision(
    model=model,
    explainer_type=ExplanationType.LIME,
    feature_names=feature_names
)

result = bbp.explain_local(X[0:1])
lime_explanation = result["explanations"]["lime"]["explanations"]
```

## Explanation Modes

### Local Explanations (Operational)

For real-time decision validation:

```python
# Single prediction with explanation
result = bbp.explain_local(X[0:1])

# Or use the convenience method
result = bbp.predict_with_explanation(X[0:1])
```

### Global Explanations (Auditing)

For model auditing and bias detection:

```python
# Analyze model behavior across dataset
audit_result = bbp.explain_global(X[:100])

# Comprehensive audit with accuracy
audit_result = bbp.audit_model(
    X[:100],
    y=y[:100],
    explanation_type=ExplanationType.SHAP
)

print("Model Accuracy:", audit_result.get("accuracy"))
print("Feature Importance:", audit_result["explanations"]["shap"]["feature_importance_ranking"])
```

## Working with Results

### Extract Key Features

```python
from blackboxpcs.utils import extract_key_features

result = bbp.explain_local(X[0:1])
top_features = extract_key_features(result, top_k=5, explainer_type="shap")

for feature in top_features["features"]:
    print(f"{feature['name']}: {feature['importance']:.4f}")
```

### Validate Explanations

```python
from blackboxpcs.utils import validate_explanation

result = bbp.explain_local(X[0:1])
validation = validate_explanation(result)

if validation["is_valid"]:
    print("Explanation is valid!")
else:
    print("Validation issues:", validation)
```

### Format for Audit Trail

```python
from blackboxpcs.utils import format_explanation_for_audit

result = bbp.explain_local(X[0:1])
audit_record = format_explanation_for_audit(result, include_raw=False)

# Save to audit log
import json
with open("audit_log.json", "a") as f:
    json.dump(audit_record, f)
```

## Complete Example

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from blackboxpcs import BlackBoxPrecision, ExplanationType
from blackboxpcs.utils import extract_key_features

# 1. Prepare data
X, y = make_classification(n_samples=200, n_features=10, random_state=42)
feature_names = [f"feature_{i}" for i in range(10)]

# 2. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 3. Initialize framework
bbp = BlackBoxPrecision(
    model=model,
    explainer_type=ExplanationType.BOTH,
    feature_names=feature_names,
    class_names=["negative", "positive"]
)

# 4. Generate explanation
result = bbp.predict_with_explanation(X[0:1])

# 5. Extract insights
top_features = extract_key_features(result, top_k=3)

print("Prediction:", result["predictions"])
print("\nTop Contributing Features:")
for feature in top_features["features"]:
    print(f"  {feature.get('name', feature['index'])}: {feature['importance']:.4f}")
```

## Next Steps

- [Core Concepts](core-concepts.md) - Understand the framework architecture
- [API Reference](api-reference.md) - Explore all available methods
- [Examples](examples.md) - See advanced usage patterns
- [Use Cases](use-cases/medical.md) - Industry-specific applications


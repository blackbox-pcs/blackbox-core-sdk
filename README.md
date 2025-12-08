# Black Box Precision Core SDK

**Unlocking High-Stakes Performance with Explainable AI**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![npm version](https://img.shields.io/npm/v/blackboxpcs)](https://www.npmjs.com/package/blackboxpcs)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

The **Black Box Precision** SDK resolves the dilemma between AI performance and interpretability. It enables you to harness maximum AI power while simultaneously integrating **Explainable Artificial Intelligence (XAI)** techniques to ensure transparency, safety, and accountability—*without* sacrificing performance.

This SDK is specifically designed for high-stakes environments where errors carry catastrophic consequences (e.g., medical diagnostics, autonomous systems, military applications, financial systems).

[Docs Live](https://docs.blackboxprecision.com)

## Key Features

- **🔬 SHAP Integration**: Theoretical gold standard for feature attribution
- **⚡ LIME Integration**: Fast, intuitive local explanations
- **🌐 Global & Local Explanations**: Support for both auditing and operational oversight
- **🛡️ High-Stakes Ready**: Built for mission-critical applications
- **📊 Comprehensive Utilities**: Tools for validation, aggregation, and audit trails

## Installation

### Via npm

The package is available on npm:

```bash
npm install blackboxpcs
```

📦 **npm package**: [https://www.npmjs.com/package/blackboxpcs](https://www.npmjs.com/package/blackboxpcs)

### Via pip (Python)

Install dependencies:

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

## Quick Start

### Basic Usage

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from blackboxpcs import BlackBoxPrecision, ExplanationType, ExplanationMode

# Train a black box model
X_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, 100)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Initialize Black Box Precision framework
bbp = BlackBoxPrecision(
    model=model,
    explainer_type=ExplanationType.BOTH,
    feature_names=[f"feature_{i}" for i in range(10)]
)

# Generate local explanation for operational oversight
X_test = np.random.rand(1, 10)
result = bbp.explain_local(X_test)

print("Prediction:", result["predictions"])
print("SHAP Explanation:", result["explanations"]["shap"])
print("LIME Explanation:", result["explanations"]["lime"])
```

### Medical Diagnostics Example (SHAP)

```python
from blackboxpcs import BlackBoxPrecision, ExplanationType

# Medical diagnosis model
diagnosis_model = load_medical_model()

bbp = BlackBoxPrecision(
    model=diagnosis_model,
    explainer_type=ExplanationType.SHAP,
    feature_names=["lesion_density", "lesion_size", "patient_age", ...],
    class_names=["benign", "malignant"]
)

# Patient data
patient_data = np.array([[0.85, 12.0, 45, ...]])

# Get prediction with explanation
result = bbp.predict_with_explanation(patient_data)

# Extract key features driving the diagnosis
from blackboxpcs.utils import extract_key_features
top_features = extract_key_features(result, top_k=5, explainer_type="shap")

print(f"Diagnosis: {result['predictions']}")
print(f"Key factors: {top_features['features']}")
```

### Autonomous Systems Example (LIME)

```python
from blackboxpcs import BlackBoxPrecision, ExplanationType

# Autonomous vehicle perception model
perception_model = load_perception_model()

bbp = BlackBoxPrecision(
    model=perception_model,
    explainer_type=ExplanationType.LIME,
    feature_names=[f"pixel_{i}" for i in range(224*224*3)]  # Image features
)

# Sensor data at decision point
sensor_data = np.array([...])  # Real-time sensor reading

# Real-time explanation for critical decision
result = bbp.explain_local(sensor_data)

# Get top contributing features
from blackboxpcs.explainers import LIMEExplainer
lime_explainer = bbp._get_lime_explainer()
top_features = lime_explainer.get_top_features(sensor_data, top_k=10)

print(f"Decision: {result['predictions']}")
print(f"Key factors: {top_features['top_features']}")
```

### Model Auditing (Global XAI)

```python
# Perform comprehensive model audit
audit_results = bbp.audit_model(
    X_train,
    y=y_train,
    explanation_type=ExplanationType.SHAP
)

print("Model Accuracy:", audit_results.get("accuracy"))
print("Feature Importance:", audit_results["explanations"]["shap"]["feature_importance_ranking"])
```

## Core Concepts

### Explanation Types

- **SHAP (SHapley Additive exPlanations)**: Provides mathematical guarantees for feature attribution. Ideal for post-mortem auditing and regulatory compliance.
- **LIME (Local Interpretable Model-agnostic Explanations)**: Fast, intuitive explanations perfect for real-time operational oversight.

### Explanation Modes

- **Local (Operational)**: Generate explanations for individual predictions in real-time
- **Global (Auditing)**: Analyze model behavior across datasets to detect biases and validate system behavior

## API Reference

### BlackBoxPrecision

Main framework class for integrating XAI with black box models.

```python
BlackBoxPrecision(
    model: Any,
    explainer_type: ExplanationType = ExplanationType.BOTH,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    **kwargs
)
```

**Key Methods:**

- `explain(X, mode, explanation_type)`: Generate explanations
- `explain_local(X)`: Generate local explanations for operational use
- `explain_global(X)`: Generate global explanations for auditing
- `predict_with_explanation(X)`: Make predictions with immediate explanations
- `audit_model(X, y)`: Perform comprehensive model auditing

### SHAPExplainer

```python
SHAPExplainer(
    model: Any,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    background_data: Optional[np.ndarray] = None,
    algorithm: str = "auto",
    **kwargs
)
```

### LIMEExplainer

```python
LIMEExplainer(
    model: Any,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    mode: str = "classification",
    num_features: int = 10,
    **kwargs
)
```

## Utilities

The SDK includes utility functions for common tasks:

- `validate_explanation()`: Validate explanation completeness
- `aggregate_explanations()`: Aggregate multiple explanations
- `format_explanation_for_audit()`: Format explanations for audit trails
- `compare_explanations()`: Compare two explanations
- `extract_key_features()`: Extract top contributing features

## Use Cases

### Medical Diagnostics
- **Challenge**: Deploying high-accuracy diagnostic AI without clinical justification
- **Solution**: SHAP provides verifiable explanations for every diagnosis
- **Impact**: Clinical trust, regulatory compliance, audit trails

### Autonomous Systems
- **Challenge**: Validating safety-critical, split-second decisions
- **Solution**: LIME provides instant explanations for real-time validation
- **Impact**: Safety verification, compliance, post-incident analysis

### Financial Systems
- **Challenge**: Explaining credit decisions and fraud detection
- **Solution**: Combined SHAP and LIME for comprehensive explanations
- **Impact**: Regulatory compliance, customer trust, bias detection

## Philosophy

**Black Box Precision** embraces the full complexity of deep AI, viewing the "Black Box" as a source of unparalleled power, not a failure of design. Our approach is built on three non-negotiable pillars:

1. **Depth of Insight**: Utilize complex models to their full capacity
2. **Trust through Results**: Generate verifiable explanations for every decision
3. **Application in Critical Fields**: Designed for high-stakes environments

## Contributing

Contributions are welcome! Please see our contributing guidelines for details.

## License

MIT License - see LICENSE file for details

## Citation

If you use Black Box Precision in your research, please cite:

```
Black Box Precision: Unlocking High-Stakes Performance with Explainable AI
The XAI Lab, 2025
```

## Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**The time to choose is now: Demand Black Box Precision.**


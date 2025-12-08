# Core Concepts

Understanding the fundamental concepts behind Black Box Precision.

## Architecture Overview

Black Box Precision provides a unified framework for integrating Explainable AI (XAI) techniques with black box machine learning models. The framework is designed around three core principles:

1. **Model Agnostic**: Works with any model that supports `predict` or `predict_proba`
2. **Dual Explainer Support**: Integrates both SHAP and LIME for comprehensive explanations
3. **Mode Flexibility**: Supports both operational (local) and auditing (global) use cases

## Explanation Types

### SHAP (SHapley Additive exPlanations)

SHAP provides the theoretical gold standard for feature attribution, calculating the fair marginal contribution of each input feature to the prediction.

**Characteristics:**
- Mathematical guarantees (efficiency, symmetry, dummy, additivity)
- Model-agnostic and consistent
- Computationally intensive for large datasets

**Best For:**
- Post-mortem auditing
- Regulatory compliance
- Understanding feature importance
- Detecting bias

### LIME (Local Interpretable Model-agnostic Explanations)

LIME provides fast, intuitive explanations by training a simple, local surrogate model around a single prediction point.

**Characteristics:**
- Fast computation
- Intuitive explanations
- Local approximations
- Less computationally intensive

**Best For:**
- Real-time operational oversight
- Split-second decision validation
- Quick feature identification
- Interactive debugging

## Explanation Modes

### Local Mode (Operational)

Local explanations focus on individual predictions, providing insights into why a specific decision was made.

**Use Cases:**
- Real-time decision validation
- Operational oversight
- Interactive debugging
- User-facing explanations

**Example:**
```python
result = bbp.explain_local(X[0:1])
# Explains why this specific instance received this prediction
```

### Global Mode (Auditing)

Global explanations analyze model behavior across datasets, detecting systemic patterns and biases.

**Use Cases:**
- Model auditing
- Bias detection
- Feature importance analysis
- Regulatory compliance

**Example:**
```python
audit_result = bbp.explain_global(X[:100])
# Analyzes model behavior across multiple instances
```

## Workflow Patterns

### Operational Workflow

```python
# 1. Initialize framework
bbp = BlackBoxPrecision(model, explainer_type=ExplanationType.LIME)

# 2. Generate real-time explanation
result = bbp.predict_with_explanation(new_instance)

# 3. Extract key features
top_features = extract_key_features(result, top_k=5)

# 4. Use for decision validation
if top_features["features"][0]["importance"] > threshold:
    approve_decision()
```

### Auditing Workflow

```python
# 1. Initialize framework
bbp = BlackBoxPrecision(model, explainer_type=ExplanationType.SHAP)

# 2. Perform comprehensive audit
audit_result = bbp.audit_model(X_train, y_train)

# 3. Analyze feature importance
feature_importance = audit_result["explanations"]["shap"]["feature_importance_ranking"]

# 4. Check for bias
if suspicious_feature in top_features:
    flag_for_review()
```

## Model Compatibility

Black Box Precision works with any model that implements:

- `predict(X)` - For classification or regression
- `predict_proba(X)` - For classification (preferred)

**Compatible Models:**
- scikit-learn models (RandomForest, XGBoost, SVM, etc.)
- TensorFlow/Keras models
- PyTorch models (with wrapper)
- Custom models (if they implement predict methods)

## Feature Names and Class Names

Providing feature names and class names enhances the quality of explanations:

```python
bbp = BlackBoxPrecision(
    model=model,
    feature_names=["age", "income", "credit_score"],
    class_names=["denied", "approved"]
)
```

**Benefits:**
- More readable explanations
- Better visualization
- Easier debugging
- Regulatory compliance

## Next Steps

- [Explanation Types](explanation-types.md) - Deep dive into SHAP and LIME
- [API Reference](api-reference.md) - Complete API documentation
- [Examples](examples.md) - Practical usage examples


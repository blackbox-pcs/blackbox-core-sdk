# Explanation Types

Deep dive into SHAP and LIME explainers.

## SHAP (SHapley Additive exPlanations)

SHAP provides mathematically grounded feature attribution based on cooperative game theory.

### Theory

SHAP values satisfy four important properties:

1. **Efficiency**: Sum of SHAP values equals prediction minus baseline
2. **Symmetry**: Features with equal contributions get equal SHAP values
3. **Dummy**: Features that don't affect output get zero SHAP values
4. **Additivity**: SHAP values are additive across features

### Algorithms

Black Box Precision automatically selects the best SHAP algorithm:

- **Tree**: For tree-based models (RandomForest, XGBoost, etc.)
- **Kernel**: For any model with background data
- **Permutation**: For any model without background data
- **Exact**: For small datasets (computationally expensive)

### Usage

```python
from blackboxpcs import BlackBoxPrecision, ExplanationType

bbp = BlackBoxPrecision(
    model=model,
    explainer_type=ExplanationType.SHAP,
    background_data=X_train[:100],  # Recommended for better explanations
    algorithm="auto"  # Auto-selects best algorithm
)

result = bbp.explain_local(X_test[0:1])
shap_values = result["explanations"]["shap"]["shap_values"]
```

### Advanced Usage

```python
from blackboxpcs.explainers import SHAPExplainer

shap_explainer = SHAPExplainer(
    model=model,
    feature_names=feature_names,
    background_data=X_train[:100],
    algorithm="kernel"
)

# Get feature attribution for specific feature
attribution = shap_explainer.get_feature_attribution(
    X_test[0:1],
    feature_idx=0,
    class_idx=1
)
```

## LIME (Local Interpretable Model-agnostic Explanations)

LIME explains individual predictions by training a local surrogate model.

### Theory

LIME works by:

1. Sampling instances around the prediction point
2. Training a simple, interpretable model on these samples
3. Using the simple model's coefficients as feature importance

### Characteristics

- **Fast**: Typically faster than SHAP
- **Local**: Explains individual predictions, not global behavior
- **Intuitive**: Easy to understand explanations
- **Flexible**: Works with tabular, text, and image data

### Usage

```python
from blackboxpcs import BlackBoxPrecision, ExplanationType

bbp = BlackBoxPrecision(
    model=model,
    explainer_type=ExplanationType.LIME,
    num_features=10  # Number of top features to show
)

result = bbp.explain_local(X_test[0:1])
lime_explanation = result["explanations"]["lime"]["explanations"]
```

### Advanced Usage

```python
from blackboxpcs.explainers import LIMEExplainer

lime_explainer = LIMEExplainer(
    model=model,
    feature_names=feature_names,
    mode="classification",
    num_features=10
)

# Get top contributing features
top_features = lime_explainer.get_top_features(
    X_test[0:1],
    top_k=5
)
```

## Choosing Between SHAP and LIME

### Use SHAP When:

- ✅ You need mathematical guarantees
- ✅ Regulatory compliance is required
- ✅ You're doing post-mortem analysis
- ✅ You have time for computation
- ✅ You need global feature importance

### Use LIME When:

- ✅ You need real-time explanations
- ✅ Speed is critical
- ✅ You're debugging individual predictions
- ✅ You want intuitive explanations
- ✅ You're doing operational oversight

### Use Both When:

- ✅ You need comprehensive coverage
- ✅ You want to validate explanations
- ✅ You're building production systems
- ✅ You need both speed and rigor

## Comparison Example

```python
from blackboxpcs import BlackBoxPrecision, ExplanationType

bbp = BlackBoxPrecision(
    model=model,
    explainer_type=ExplanationType.BOTH
)

result = bbp.explain_local(X_test[0:1])

# Compare SHAP and LIME explanations
shap_features = result["explanations"]["shap"]["feature_importance_ranking"][:5]
lime_features = sorted(
    result["explanations"]["lime"]["explanations"]["feature_weights"].items(),
    key=lambda x: abs(x[1]),
    reverse=True
)[:5]

print("Top 5 Features (SHAP):", shap_features)
print("Top 5 Features (LIME):", lime_features)
```

## Next Steps

- [API Reference](api-reference.md) - Complete API documentation
- [Examples](examples.md) - See SHAP and LIME in action
- [Use Cases](use-cases/medical.md) - Industry-specific applications


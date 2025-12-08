# Examples

Real-world examples demonstrating Black Box Precision capabilities.

## Basic Classification

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from blackboxpcs import BlackBoxPrecision, ExplanationType

# Generate data
X, y = make_classification(n_samples=200, n_features=10, random_state=42)
feature_names = [f"feature_{i}" for i in range(10)]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Initialize framework
bbp = BlackBoxPrecision(
    model=model,
    explainer_type=ExplanationType.BOTH,
    feature_names=feature_names,
    class_names=["negative", "positive"]
)

# Generate explanation
result = bbp.predict_with_explanation(X[0:1])

print("Prediction:", result["predictions"])
print("SHAP Values:", result["explanations"]["shap"]["shap_values"])
print("LIME Weights:", result["explanations"]["lime"]["explanations"]["feature_weights"])
```

## Model Auditing

```python
from blackboxpcs import BlackBoxPrecision, ExplanationType

bbp = BlackBoxPrecision(
    model=model,
    explainer_type=ExplanationType.SHAP,
    feature_names=feature_names
)

# Comprehensive audit
audit_result = bbp.audit_model(
    X_train[:100],
    y=y_train[:100]
)

print("Model Accuracy:", audit_result.get("accuracy"))
print("\nFeature Importance Ranking:")
for feature, importance in audit_result["explanations"]["shap"]["feature_importance_ranking"][:10]:
    print(f"  {feature}: {importance:.4f}")
```

## Batch Explanations

```python
# Explain multiple instances
results = []
for instance in X_test[:10]:
    result = bbp.explain_local(instance.reshape(1, -1))
    results.append(result)

# Aggregate explanations
from blackboxpcs.utils import aggregate_explanations

aggregated = aggregate_explanations(results, method="mean")
print("Aggregated SHAP Values:", aggregated["shap_values"])
```

## Feature Extraction

```python
from blackboxpcs.utils import extract_key_features

result = bbp.explain_local(X_test[0:1])

# Extract top features from SHAP
top_shap = extract_key_features(result, top_k=5, explainer_type="shap")
print("Top SHAP Features:")
for feature in top_shap["features"]:
    print(f"  {feature.get('name', feature['index'])}: {feature['importance']:.4f}")

# Extract top features from LIME
top_lime = extract_key_features(result, top_k=5, explainer_type="lime")
print("\nTop LIME Features:")
for feature in top_lime["features"]:
    print(f"  {feature['name']}: {feature['importance']:.4f}")
```

## Explanation Validation

```python
from blackboxpcs.utils import validate_explanation, compare_explanations

# Validate single explanation
result = bbp.explain_local(X_test[0:1])
validation = validate_explanation(result)

if validation["is_valid"]:
    print("✓ Explanation is valid")
else:
    print("✗ Validation issues:", validation)

# Compare two explanations
result1 = bbp.explain_local(X_test[0:1])
result2 = bbp.explain_local(X_test[1:2])

comparison = compare_explanations(result1, result2, metric="cosine")
print("SHAP Similarity:", comparison.get("shap_similarity"))
print("Prediction Match:", comparison.get("prediction_match"))
```

## Audit Trail Generation

```python
from blackboxpcs.utils import format_explanation_for_audit
import json
from datetime import datetime

result = bbp.explain_local(X_test[0:1])
audit_record = format_explanation_for_audit(result, include_raw=False)
audit_record["timestamp"] = datetime.now().isoformat()

# Save to audit log
with open("audit_log.json", "a") as f:
    json.dump(audit_record, f)
    f.write("\n")
```

## Custom Model Integration

```python
import numpy as np
from blackboxpcs import BlackBoxPrecision, ExplanationType

class CustomModel:
    def __init__(self):
        self.weights = np.random.randn(10)
    
    def predict(self, X):
        return (X @ self.weights > 0).astype(int)
    
    def predict_proba(self, X):
        probs = 1 / (1 + np.exp(-X @ self.weights))
        return np.column_stack([1 - probs, probs])

# Use with custom model
custom_model = CustomModel()
bbp = BlackBoxPrecision(
    model=custom_model,
    explainer_type=ExplanationType.BOTH
)

result = bbp.explain_local(X_test[0:1])
```

## Regression Example

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

# Generate regression data
X, y = make_regression(n_samples=200, n_features=10, random_state=42)

# Train regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Initialize with LIME (better for regression)
bbp = BlackBoxPrecision(
    model=model,
    explainer_type=ExplanationType.LIME,
    feature_names=[f"feature_{i}" for i in range(10)]
)

# Explain regression prediction
result = bbp.explain_local(X[0:1].reshape(1, -1))
print("Prediction:", result["predictions"])
print("Feature Contributions:", result["explanations"]["lime"]["explanations"]["feature_weights"])
```

## Next Steps

- [Use Cases](use-cases/medical.md) - Industry-specific examples
- [API Reference](api-reference.md) - Complete API documentation
- [Contributing](contributing.md) - Add your own examples


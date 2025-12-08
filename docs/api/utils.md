# Utility Functions

Helper functions for validation, aggregation, and workflow management.

## Functions

### `validate_explanation()`

Validate an explanation for completeness and consistency.

```python
def validate_explanation(
    explanation: Dict[str, Any],
    prediction: Optional[np.ndarray] = None
) -> Dict[str, bool]:
```

**Parameters:**
- `explanation`: Explanation dictionary from explainer
- `prediction`: Optional prediction to validate against

**Returns:**
Dictionary with validation results

**Example:**
```python
from blackboxpcs.utils import validate_explanation

result = bbp.explain_local(X_test[0:1])
validation = validate_explanation(result)

if validation["is_valid"]:
    print("✓ Explanation is valid")
```

### `aggregate_explanations()`

Aggregate multiple explanations for global analysis.

```python
def aggregate_explanations(
    explanations: List[Dict[str, Any]],
    method: str = "mean"
) -> Dict[str, Any]:
```

**Parameters:**
- `explanations`: List of explanation dictionaries
- `method`: Aggregation method (`"mean"`, `"median"`, `"max"`, `"min"`)

**Returns:**
Aggregated explanation dictionary

**Example:**
```python
from blackboxpcs.utils import aggregate_explanations

results = [bbp.explain_local(X[i:i+1]) for i in range(10)]
aggregated = aggregate_explanations(results, method="mean")
```

### `format_explanation_for_audit()`

Format explanation for audit trail and regulatory compliance.

```python
def format_explanation_for_audit(
    explanation: Dict[str, Any],
    include_raw: bool = False
) -> Dict[str, Any]:
```

**Parameters:**
- `explanation`: Explanation dictionary
- `include_raw`: Whether to include raw explanation data

**Returns:**
Formatted explanation suitable for audit logs

**Example:**
```python
from blackboxpcs.utils import format_explanation_for_audit
import json

result = bbp.explain_local(X_test[0:1])
audit_record = format_explanation_for_audit(result, include_raw=False)

with open("audit_log.json", "a") as f:
    json.dump(audit_record, f)
```

### `compare_explanations()`

Compare two explanations to measure consistency.

```python
def compare_explanations(
    explanation1: Dict[str, Any],
    explanation2: Dict[str, Any],
    metric: str = "cosine"
) -> Dict[str, Any]:
```

**Parameters:**
- `explanation1`: First explanation
- `explanation2`: Second explanation
- `metric`: Comparison metric (`"cosine"`, `"euclidean"`, `"manhattan"`)

**Returns:**
Dictionary with comparison results

**Example:**
```python
from blackboxpcs.utils import compare_explanations

result1 = bbp.explain_local(X_test[0:1])
result2 = bbp.explain_local(X_test[1:2])

comparison = compare_explanations(result1, result2, metric="cosine")
print("Similarity:", comparison.get("shap_similarity"))
```

### `extract_key_features()`

Extract top contributing features from an explanation.

```python
def extract_key_features(
    explanation: Dict[str, Any],
    top_k: int = 5,
    explainer_type: Optional[str] = None
) -> Dict[str, Any]:
```

**Parameters:**
- `explanation`: Explanation dictionary
- `top_k`: Number of top features to extract
- `explainer_type`: Specific explainer to use (`"shap"` or `"lime"`)

**Returns:**
Dictionary with top features and their contributions

**Example:**
```python
from blackboxpcs.utils import extract_key_features

result = bbp.explain_local(X_test[0:1])
top_features = extract_key_features(result, top_k=5, explainer_type="shap")

for feature in top_features["features"]:
    print(f"{feature.get('name', feature['index'])}: {feature['importance']:.4f}")
```

## Examples

### Complete Workflow

```python
from blackboxpcs.utils import (
    validate_explanation,
    extract_key_features,
    format_explanation_for_audit
)

# Generate explanation
result = bbp.explain_local(X_test[0:1])

# Validate
validation = validate_explanation(result)
assert validation["is_valid"], "Invalid explanation"

# Extract key features
top_features = extract_key_features(result, top_k=5)

# Format for audit
audit_record = format_explanation_for_audit(result)
```


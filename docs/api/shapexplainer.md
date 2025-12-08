# SHAPExplainer

SHAP (SHapley Additive exPlanations) Explainer for feature attribution.

## Class Definition

```python
class SHAPExplainer(BaseExplainer):
    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        background_data: Optional[np.ndarray] = None,
        algorithm: str = "auto",
        **kwargs
    ):
        ...
```

## Parameters

### `model`
The black box model to explain.

**Type:** Any  
**Required:** Yes

### `feature_names`
Optional feature names for better interpretability.

**Type:** `Optional[List[str]]`  
**Default:** `None`

### `class_names`
Optional class names for classification tasks.

**Type:** `Optional[List[str]]`  
**Default:** `None`

### `background_data`
Background dataset for SHAP (recommended for better explanations).

**Type:** `Optional[np.ndarray]`  
**Default:** `None`

### `algorithm`
SHAP algorithm to use.

**Type:** `str`  
**Default:** `"auto"`  
**Options:**
- `"auto"` - Auto-selects best algorithm
- `"tree"` - For tree-based models
- `"kernel"` - For any model with background data
- `"permutation"` - For any model without background data
- `"exact"` - Exact SHAP (computationally expensive)
- `"sampling"` - Sampling-based approximation

## Methods

### `explain()`

Generate SHAP explanations.

```python
def explain(
    self,
    X: np.ndarray,
    mode: ExplanationMode = ExplanationMode.LOCAL,
    **kwargs
) -> Dict[str, Any]:
```

**Parameters:**
- `X`: Input data to explain
- `mode`: Explanation mode (`GLOBAL` or `LOCAL`)
- `**kwargs`: Additional SHAP parameters

**Returns:**
Dictionary containing SHAP values and metadata

**Example:**
```python
from blackboxpcs.explainers import SHAPExplainer

shap_explainer = SHAPExplainer(
    model=model,
    background_data=X_train[:100],
    algorithm="kernel"
)

result = shap_explainer.explain(X_test[0:1])
shap_values = result["shap_values"]
```

### `get_feature_attribution()`

Get feature attribution for specific features/classes.

```python
def get_feature_attribution(
    self,
    X: np.ndarray,
    feature_idx: Optional[int] = None,
    class_idx: Optional[int] = None
) -> Dict[str, Any]:
```

**Parameters:**
- `X`: Input data
- `feature_idx`: Specific feature index (None for all)
- `class_idx`: Specific class index for classification (None for all)

**Returns:**
Dictionary with feature attributions

**Example:**
```python
attribution = shap_explainer.get_feature_attribution(
    X_test[0:1],
    feature_idx=0,
    class_idx=1
)
```

## Return Format

```python
{
    "shap_values": np.ndarray,
    "base_values": np.ndarray,
    "mode": str,
    "algorithm": str,
    "feature_names": List[str],
    "class_names": List[str],
    "feature_importance": List[float],  # Global mode only
    "feature_importance_ranking": List[Tuple[str, float]]  # Global mode only
}
```

## Examples

### Basic Usage

```python
from blackboxpcs.explainers import SHAPExplainer

shap_explainer = SHAPExplainer(
    model=model,
    feature_names=["age", "income", "credit_score"],
    background_data=X_train[:100]
)

result = shap_explainer.explain(X_test[0:1])
```

### Global Feature Importance

```python
result = shap_explainer.explain(
    X_train[:100],
    mode=ExplanationMode.GLOBAL
)

print("Feature Importance Ranking:")
for feature, importance in result["feature_importance_ranking"][:10]:
    print(f"  {feature}: {importance:.4f}")
```

### Specific Feature Attribution

```python
attribution = shap_explainer.get_feature_attribution(
    X_test[0:1],
    feature_idx=0  # First feature
)

print("Attribution:", attribution["attributions"])
```


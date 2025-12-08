# LIMEExplainer

LIME (Local Interpretable Model-agnostic Explanations) Explainer for fast local explanations.

## Class Definition

```python
class LIMEExplainer(BaseExplainer):
    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        mode: str = "classification",
        num_features: int = 10,
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

### `mode`
Task mode.

**Type:** `str`  
**Default:** `"classification"`  
**Options:**
- `"classification"` - For classification tasks
- `"regression"` - For regression tasks

### `num_features`
Number of top features to show in explanation.

**Type:** `int`  
**Default:** `10`

## Methods

### `explain()`

Generate LIME explanations.

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
- `**kwargs`: Additional LIME parameters (e.g., `num_features`)

**Returns:**
Dictionary containing LIME explanations

**Example:**
```python
from blackboxpcs.explainers import LIMEExplainer

lime_explainer = LIMEExplainer(
    model=model,
    num_features=10,
    mode="classification"
)

result = lime_explainer.explain(X_test[0:1])
```

### `get_top_features()`

Get top contributing features for the prediction.

```python
def get_top_features(
    self,
    X: np.ndarray,
    top_k: Optional[int] = None
) -> Dict[str, Any]:
```

**Parameters:**
- `X`: Input data (single instance)
- `top_k`: Number of top features to return (None for all)

**Returns:**
Dictionary with top features and their contributions

**Example:**
```python
top_features = lime_explainer.get_top_features(X_test[0:1], top_k=5)

for feature_name, weight in top_features["top_features"]:
    print(f"{feature_name}: {weight:.4f}")
```

## Return Format

```python
{
    "explanations": Dict,  # Single instance or list for batch
    "mode": str,
    "num_features": int,
    "feature_names": List[str],
    "class_names": List[str],
    "feature_importance": Dict,  # Global mode only
    "feature_importance_ranking": List[Tuple[str, float]]  # Global mode only
}
```

## Examples

### Basic Usage

```python
from blackboxpcs.explainers import LIMEExplainer

lime_explainer = LIMEExplainer(
    model=model,
    feature_names=["age", "income", "credit_score"],
    num_features=10
)

result = lime_explainer.explain(X_test[0:1])
feature_weights = result["explanations"]["feature_weights"]
```

### Get Top Features

```python
top_features = lime_explainer.get_top_features(X_test[0:1], top_k=5)

print("Top Contributing Features:")
for feature_name, weight in top_features["top_features"]:
    print(f"  {feature_name}: {weight:.4f}")
```

### Regression Mode

```python
lime_explainer = LIMEExplainer(
    model=regression_model,
    mode="regression",
    num_features=10
)

result = lime_explainer.explain(X_test[0:1])
```


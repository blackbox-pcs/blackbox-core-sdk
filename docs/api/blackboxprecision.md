# BlackBoxPrecision

Main framework class for integrating XAI with black box models.

## Class Definition

```python
class BlackBoxPrecision:
    def __init__(
        self,
        model: Any,
        explainer_type: ExplanationType = ExplanationType.BOTH,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        **kwargs
    ):
        ...
```

## Parameters

### `model`
The black box model to explain. Must support `predict()` or `predict_proba()` methods.

**Type:** Any  
**Required:** Yes

### `explainer_type`
Type of explainer(s) to use.

**Type:** `ExplanationType`  
**Default:** `ExplanationType.BOTH`  
**Options:**
- `ExplanationType.SHAP` - SHAP only
- `ExplanationType.LIME` - LIME only
- `ExplanationType.BOTH` - Both SHAP and LIME

### `feature_names`
Optional list of feature names for better interpretability.

**Type:** `Optional[List[str]]`  
**Default:** `None`

### `class_names`
Optional list of class names for classification tasks.

**Type:** `Optional[List[str]]`  
**Default:** `None`

### `**kwargs`
Additional arguments passed to explainers (e.g., `background_data`, `algorithm`, `num_features`).

## Methods

### `explain()`

Generate explanations for predictions.

```python
def explain(
    self,
    X: np.ndarray,
    mode: ExplanationMode = ExplanationMode.LOCAL,
    explanation_type: Optional[ExplanationType] = None,
    X_background: Optional[np.ndarray] = None,
    **kwargs
) -> Dict[str, Any]:
```

**Parameters:**
- `X`: Input data to explain (single instance or batch)
- `mode`: Explanation mode (`GLOBAL` for auditing, `LOCAL` for operational)
- `explanation_type`: Override default explainer type
- `X_background`: Background data for SHAP (optional)
- `**kwargs`: Additional arguments for explainers

**Returns:**
Dictionary containing explanations and metadata

**Example:**
```python
result = bbp.explain(X_test, mode=ExplanationMode.GLOBAL)
```

### `explain_local()`

Generate local explanations for operational oversight.

```python
def explain_local(
    self,
    X: np.ndarray,
    explanation_type: Optional[ExplanationType] = None,
    **kwargs
) -> Dict[str, Any]:
```

**Parameters:**
- `X`: Input data to explain
- `explanation_type`: Override default explainer type
- `**kwargs`: Additional arguments for explainers

**Returns:**
Dictionary containing local explanations

**Example:**
```python
result = bbp.explain_local(X_test[0:1])
```

### `explain_global()`

Generate global explanations for auditing and development.

```python
def explain_global(
    self,
    X: np.ndarray,
    explanation_type: Optional[ExplanationType] = None,
    **kwargs
) -> Dict[str, Any]:
```

**Parameters:**
- `X`: Input data to explain (typically training/validation set)
- `explanation_type`: Override default explainer type
- `**kwargs`: Additional arguments for explainers

**Returns:**
Dictionary containing global explanations

**Example:**
```python
result = bbp.explain_global(X_train[:100])
```

### `predict_with_explanation()`

Make predictions with immediate explanations.

```python
def predict_with_explanation(
    self,
    X: np.ndarray,
    explanation_type: Optional[ExplanationType] = None,
    **kwargs
) -> Dict[str, Any]:
```

**Parameters:**
- `X`: Input data
- `explanation_type`: Override default explainer type
- `**kwargs`: Additional arguments for explainers

**Returns:**
Dictionary with predictions and explanations

**Example:**
```python
result = bbp.predict_with_explanation(X_test[0:1])
```

### `audit_model()`

Perform comprehensive model auditing using global XAI.

```python
def audit_model(
    self,
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    explanation_type: Optional[ExplanationType] = None,
    **kwargs
) -> Dict[str, Any]:
```

**Parameters:**
- `X`: Dataset for auditing (typically training/validation set)
- `y`: Optional ground truth labels for validation
- `explanation_type`: Override default explainer type
- `**kwargs`: Additional arguments for explainers

**Returns:**
Dictionary containing audit results and global explanations

**Example:**
```python
audit_result = bbp.audit_model(X_train, y_train)
print("Accuracy:", audit_result.get("accuracy"))
```

## Return Format

All explanation methods return a dictionary with the following structure:

```python
{
    "predictions": np.ndarray,  # Model predictions
    "mode": str,                 # "local" or "global"
    "explanations": {
        "shap": {                # If SHAP is enabled
            "shap_values": np.ndarray,
            "base_values": np.ndarray,
            "algorithm": str,
            "feature_importance": List[float],  # Global mode only
            "feature_importance_ranking": List[Tuple[str, float]]  # Global mode only
        },
        "lime": {                # If LIME is enabled
            "explanations": Dict,
            "feature_importance": Dict,  # Global mode only
            "feature_importance_ranking": List[Tuple[str, float]]  # Global mode only
        }
    }
}
```

## Examples

### Basic Usage

```python
from blackboxpcs import BlackBoxPrecision, ExplanationType

bbp = BlackBoxPrecision(
    model=model,
    explainer_type=ExplanationType.BOTH,
    feature_names=["age", "income", "credit_score"]
)

result = bbp.explain_local(X_test[0:1])
```

### Model Auditing

```python
audit_result = bbp.audit_model(
    X_train[:100],
    y=y_train[:100],
    explanation_type=ExplanationType.SHAP
)

print("Model Accuracy:", audit_result.get("accuracy"))
print("Top Features:", audit_result["explanations"]["shap"]["feature_importance_ranking"][:5])
```

### Custom Background Data

```python
result = bbp.explain(
    X_test[0:1],
    X_background=X_train[:100],  # Custom background for SHAP
    explanation_type=ExplanationType.SHAP
)
```


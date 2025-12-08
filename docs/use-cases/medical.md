# Medical Diagnostics Use Case

Using Black Box Precision for medical diagnosis with SHAP explanations.

## Challenge

Deploying high-accuracy diagnostic AI without clinical justification is a critical challenge in healthcare. Medical professionals need to understand why a model made a specific diagnosis to:

- Build clinical trust
- Ensure regulatory compliance
- Maintain audit trails
- Validate model decisions

## Solution

Black Box Precision with SHAP provides verifiable explanations for every diagnosis, enabling transparent AI-assisted medical decision-making.

## Implementation

### Setup

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from blackboxpcs import BlackBoxPrecision, ExplanationType

# Load medical model (example)
# In practice, this would be your trained diagnostic model
diagnosis_model = load_medical_model()

# Initialize with SHAP for regulatory compliance
bbp = BlackBoxPrecision(
    model=diagnosis_model,
    explainer_type=ExplanationType.SHAP,
    feature_names=[
        "lesion_density",
        "lesion_size",
        "patient_age",
        "family_history",
        "biomarker_1",
        "biomarker_2"
    ],
    class_names=["benign", "malignant"]
)
```

### Patient Diagnosis with Explanation

```python
# Patient data
patient_data = np.array([[
    0.85,  # lesion_density
    12.0,  # lesion_size (mm)
    45,    # patient_age
    1,     # family_history (yes)
    0.7,   # biomarker_1
    0.9    # biomarker_2
]])

# Get prediction with explanation
result = bbp.predict_with_explanation(patient_data)

# Extract key features driving the diagnosis
from blackboxpcs.utils import extract_key_features

top_features = extract_key_features(result, top_k=5, explainer_type="shap")

print(f"Diagnosis: {result['predictions']}")
print("\nKey Factors:")
for feature in top_features["features"]:
    print(f"  {feature['name']}: {feature['importance']:.4f}")
```

### Clinical Audit Trail

```python
from blackboxpcs.utils import format_explanation_for_audit
from datetime import datetime
import json

# Generate explanation
result = bbp.predict_with_explanation(patient_data)

# Format for audit trail
audit_record = format_explanation_for_audit(result, include_raw=False)
audit_record.update({
    "timestamp": datetime.now().isoformat(),
    "patient_id": "P12345",
    "clinician": "Dr. Smith"
})

# Save to audit log
with open("clinical_audit_log.json", "a") as f:
    json.dump(audit_record, f)
    f.write("\n")
```

### Model Validation

```python
# Perform comprehensive model audit
audit_results = bbp.audit_model(
    X_train,
    y=y_train,
    explanation_type=ExplanationType.SHAP
)

print("Model Accuracy:", audit_results.get("accuracy"))
print("\nFeature Importance Ranking:")
for feature, importance in audit_results["explanations"]["shap"]["feature_importance_ranking"]:
    print(f"  {feature}: {importance:.4f}")

# Check for bias
suspicious_features = ["patient_age", "gender"]
for feature, importance in audit_results["explanations"]["shap"]["feature_importance_ranking"]:
    if feature in suspicious_features and importance > threshold:
        print(f"⚠️  Warning: High importance for {feature} - potential bias")
```

## Impact

- ✅ **Clinical Trust**: Doctors understand model decisions
- ✅ **Regulatory Compliance**: Audit trails for FDA/regulatory bodies
- ✅ **Bias Detection**: Identify discriminatory features
- ✅ **Model Validation**: Verify model behavior before deployment

## Best Practices

1. **Use SHAP for Medical Applications**: Provides mathematical guarantees needed for regulatory compliance
2. **Maintain Audit Trails**: Log all explanations for compliance
3. **Validate Feature Importance**: Ensure model uses clinically relevant features
4. **Monitor for Bias**: Regularly audit model for discriminatory patterns

## Related

- [Autonomous Systems](autonomous.md) - Real-time decision validation
- [Financial Systems](financial.md) - Regulatory compliance
- [API Reference](../api-reference.md) - Complete API documentation


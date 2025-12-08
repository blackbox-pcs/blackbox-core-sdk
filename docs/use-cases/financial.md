# Financial Systems Use Case

Using Black Box Precision for credit decisions and fraud detection.

## Challenge

Explaining credit decisions and fraud detection requires:

- Regulatory compliance (e.g., Fair Lending Act)
- Customer trust and transparency
- Bias detection
- Audit trails

## Solution

Black Box Precision with combined SHAP and LIME provides comprehensive explanations for financial decisions, enabling regulatory compliance and customer trust.

## Implementation

### Credit Decision Explanation

```python
import numpy as np
from blackboxpcs import BlackBoxPrecision, ExplanationType

# Credit scoring model
credit_model = load_credit_model()

# Initialize with both explainers
bbp = BlackBoxPrecision(
    model=credit_model,
    explainer_type=ExplanationType.BOTH,
    feature_names=[
        "credit_score",
        "income",
        "debt_to_income",
        "employment_years",
        "loan_amount",
        "loan_term"
    ],
    class_names=["denied", "approved"]
)
```

### Credit Decision with Explanation

```python
# Applicant data
applicant_data = np.array([[
    720,   # credit_score
    75000, # income
    0.35,  # debt_to_income
    5,     # employment_years
    25000, # loan_amount
    60     # loan_term (months)
]])

# Get decision with explanation
result = bbp.predict_with_explanation(applicant_data)

# Extract key factors
from blackboxpcs.utils import extract_key_features

top_features = extract_key_features(result, top_k=5)

print(f"Decision: {result['predictions']}")
print("\nKey Factors:")
for feature in top_features["features"]:
    print(f"  {feature.get('name', feature['index'])}: {feature['importance']:.4f}")

# Generate customer-facing explanation
customer_explanation = format_customer_explanation(result, top_features)
send_to_customer(customer_explanation)
```

### Bias Detection

```python
# Audit model for bias
audit_result = bbp.audit_model(
    X_train,
    y=y_train,
    explanation_type=ExplanationType.SHAP
)

# Check for protected class bias
protected_features = ["age", "gender", "race", "zip_code"]
feature_importance = audit_result["explanations"]["shap"]["feature_importance_ranking"]

print("Bias Analysis:")
for feature, importance in feature_importance:
    if feature in protected_features:
        if importance > bias_threshold:
            print(f"⚠️  WARNING: High importance for protected feature {feature}")
            flag_for_review()
```

### Fraud Detection

```python
# Fraud detection model
fraud_model = load_fraud_model()

bbp_fraud = BlackBoxPrecision(
    model=fraud_model,
    explainer_type=ExplanationType.LIME,  # Fast for real-time
    feature_names=transaction_features
)

# Real-time transaction analysis
transaction = np.array([...])  # Transaction features

result = bbp_fraud.explain_local(transaction)

# Check if fraud indicators are present
top_features = extract_key_features(result, top_k=5)
fraud_indicators = ["unusual_location", "high_amount", "time_of_day"]

for feature in top_features["features"]:
    if feature["name"] in fraud_indicators:
        if feature["importance"] > threshold:
            flag_transaction_for_review()
```

### Regulatory Compliance

```python
from blackboxpcs.utils import format_explanation_for_audit
from datetime import datetime
import json

# Generate explanation for credit decision
result = bbp.predict_with_explanation(applicant_data)

# Format for regulatory audit
audit_record = format_explanation_for_audit(result, include_raw=True)
audit_record.update({
    "timestamp": datetime.now().isoformat(),
    "applicant_id": "A12345",
    "decision": result["predictions"],
    "regulatory_compliant": True
})

# Save to compliance log
with open("compliance_log.json", "a") as f:
    json.dump(audit_record, f)
    f.write("\n")
```

## Impact

- ✅ **Regulatory Compliance**: Meet Fair Lending and other regulations
- ✅ **Customer Trust**: Transparent explanations build confidence
- ✅ **Bias Detection**: Identify and mitigate discriminatory patterns
- ✅ **Risk Management**: Understand fraud detection decisions

## Best Practices

1. **Use Both Explainers**: SHAP for compliance, LIME for real-time
2. **Monitor for Bias**: Regularly audit for protected class discrimination
3. **Maintain Audit Trails**: Keep all explanations for regulatory review
4. **Customer Communication**: Provide clear, understandable explanations

## Related

- [Medical Diagnostics](medical.md) - Regulatory compliance
- [Autonomous Systems](autonomous.md) - Real-time decisions
- [API Reference](../api-reference.md) - Complete API documentation


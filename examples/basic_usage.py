"""
Basic Usage Example for Black Box Precision SDK

Demonstrates core functionality with a simple classification task.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from blackboxpcs import BlackBoxPrecision, ExplanationType, ExplanationMode

def main():
    print("=" * 60)
    print("Black Box Precision - Basic Usage Example")
    print("=" * 60)
    
    # Generate synthetic dataset
    print("\n1. Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    class_names = ["class_0", "class_1"]
    
    # Train a black box model
    print("2. Training black box model (Random Forest)...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    print(f"   Model accuracy: {model.score(X, y):.4f}")
    
    # Initialize Black Box Precision framework
    print("\n3. Initializing Black Box Precision framework...")
    bbp = BlackBoxPrecision(
        model=model,
        explainer_type=ExplanationType.BOTH,
        feature_names=feature_names,
        class_names=class_names
    )
    
    # Generate local explanation for a single instance
    print("\n4. Generating local explanation (operational oversight)...")
    test_instance = X[0:1]  # Single instance
    
    result = bbp.explain_local(
        test_instance,
        explanation_type=ExplanationType.BOTH
    )
    
    print(f"\n   Prediction: {result['predictions']}")
    print(f"   Mode: {result['mode']}")
    
    # Display SHAP explanation
    if "shap" in result["explanations"]:
        shap_exp = result["explanations"]["shap"]
        print(f"\n   SHAP Explanation:")
        print(f"     Algorithm: {shap_exp.get('algorithm')}")
        if "feature_importance" in shap_exp:
            print(f"     Top 3 Features:")
            for i, (feature, importance) in enumerate(
                shap_exp["feature_importance_ranking"][:3], 1
            ):
                print(f"       {i}. {feature}: {importance:.4f}")
    
    # Display LIME explanation
    if "lime" in result["explanations"]:
        lime_exp = result["explanations"]["lime"]
        print(f"\n   LIME Explanation:")
        if isinstance(lime_exp["explanations"], dict):
            exp = lime_exp["explanations"]
            print(f"     Top 3 Features:")
            sorted_features = sorted(
                exp["feature_weights"].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3]
            for i, (feature, weight) in enumerate(sorted_features, 1):
                print(f"       {i}. {feature}: {weight:.4f}")
    
    # Generate global explanation for auditing
    print("\n5. Generating global explanation (model auditing)...")
    global_result = bbp.explain_global(
        X[:50],  # Sample of training data
        explanation_type=ExplanationType.SHAP
    )
    
    if "shap" in global_result["explanations"]:
        shap_global = global_result["explanations"]["shap"]
        print(f"\n   Global Feature Importance (Top 5):")
        for i, (feature, importance) in enumerate(
            shap_global["feature_importance_ranking"][:5], 1
        ):
            print(f"     {i}. {feature}: {importance:.4f}")
    
    # Model auditing
    print("\n6. Performing comprehensive model audit...")
    audit_result = bbp.audit_model(
        X[:100],
        y=y[:100],
        explanation_type=ExplanationType.SHAP
    )
    
    print(f"\n   Audit Results:")
    print(f"     Accuracy: {audit_result.get('accuracy', 'N/A'):.4f}")
    print(f"     Mode: {audit_result['mode']}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()



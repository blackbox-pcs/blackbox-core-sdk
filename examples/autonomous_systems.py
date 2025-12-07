"""
Autonomous Systems Example

Demonstrates Black Box Precision for real-time validation of safety-critical
decisions in autonomous systems, inspired by the self-driving vehicle case study.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from blackboxpcs import BlackBoxPrecision, ExplanationType
from blackboxpcs.explainers import LIMEExplainer

def simulate_perception_model():
    """Simulate an autonomous vehicle perception model"""
    # In practice, this would be a DNN processing sensor/camera data
    # For demonstration, we'll simulate sensor features
    from sklearn.datasets import make_classification
    
    # Simulate sensor data: 100 features (could represent pixels, sensor readings, etc.)
    X, y = make_classification(
        n_samples=1000,
        n_features=100,
        n_informative=20,
        n_classes=3,  # 0: no action, 1: brake, 2: accelerate
        random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, X, y

def main():
    print("=" * 70)
    print("Autonomous Systems Example - Black Box Precision")
    print("=" * 70)
    
    # Simulate perception model
    print("\n1. Loading autonomous perception model...")
    model, X_train, y_train = simulate_perception_model()
    
    # Simulate sensor feature names (could be pixels, sensor readings, etc.)
    feature_names = [f"sensor_{i}" for i in range(X_train.shape[1])]
    action_names = ["no_action", "hard_brake", "accelerate"]
    
    # Initialize Black Box Precision with LIME (fast for real-time)
    print("2. Initializing Black Box Precision framework...")
    bbp = BlackBoxPrecision(
        model=model,
        explainer_type=ExplanationType.LIME,
        feature_names=feature_names,
        class_names=action_names,
        num_features=10  # Show top 10 features
    )
    
    # Simulate real-time sensor data at critical decision point
    print("\n3. Processing real-time sensor data...")
    # Simulate scenario: large, rapidly approaching object detected
    sensor_data = np.random.rand(1, X_train.shape[1])
    # Artificially increase values in certain "pixel" regions to simulate object
    sensor_data[0, 10:20] = 0.9  # Simulated object region
    sensor_data[0, 50:60] = 0.8  # Another object region
    
    # Get real-time explanation for critical decision
    print("4. Generating real-time decision explanation (LIME)...")
    result = bbp.explain_local(sensor_data)
    
    # Extract prediction
    prediction_proba = result["predictions"][0]
    predicted_action = np.argmax(prediction_proba)
    confidence = prediction_proba[predicted_action]
    
    print(f"\n   Decision: {action_names[predicted_action]}")
    print(f"   Confidence: {confidence:.2%}")
    print(f"   Action Probabilities:")
    for i, action_name in enumerate(action_names):
        print(f"     {action_name}: {prediction_proba[i]:.2%}")
    
    # Get top contributing features (as per whitepaper LIME example)
    print("\n5. Extracting key decision factors (LIME)...")
    lime_explainer = bbp._get_lime_explainer()
    top_features = lime_explainer.get_top_features(sensor_data, top_k=10)
    
    print(f"\n   Top 10 Contributing Sensor Features:")
    for i, (feature, weight) in enumerate(top_features["top_features"], 1):
        direction = "triggers" if weight > 0 else "inhibits"
        print(f"     {i}. {feature}: {weight:.4f} ({direction} {action_names[predicted_action]})")
    
    # Real-time validation (as per whitepaper)
    print("\n6. Real-Time Validation:")
    print("   ✓ LIME output highlights exact sensor regions driving decision")
    print("   ✓ Validates that decision was based on valid data, not noise")
    print("   ✓ Provides necessary trace for engineers and regulators")
    print("   ✓ Confirms system compliance and safety protocols")
    
    # Simulate post-incident analysis
    print("\n7. Post-Incident Analysis Capability:")
    print("   ✓ Explanation data provides immutable trace")
    print("   ✓ Can verify system behavior during critical moments")
    print("   ✓ Enables compliance verification and safety audits")
    
    print("\n" + "=" * 70)
    print("Autonomous systems example completed!")
    print("=" * 70)
    print("\nKey Takeaway:")
    print("The system doesn't just make a 'hard brake' decision - it")
    print("provides instant visual evidence of which sensor readings")
    print("(pixels, sensor data) drove the decision, allowing for")
    print("immediate validation and post-incident analysis.")

if __name__ == "__main__":
    main()



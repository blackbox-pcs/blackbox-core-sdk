"""
Utility Functions for Black Box Precision

Helper functions for visualization, validation, and workflow management.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from .core import ExplanationType, ExplanationMode


def validate_explanation(
    explanation: Dict[str, Any],
    prediction: Optional[np.ndarray] = None
) -> Dict[str, bool]:
    """
    Validate an explanation for completeness and consistency.
    
    Args:
        explanation: Explanation dictionary from explainer
        prediction: Optional prediction to validate against
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        "has_explanation": "explanations" in explanation or "shap_values" in explanation,
        "has_predictions": "predictions" in explanation,
        "has_mode": "mode" in explanation,
    }
    
    if prediction is not None and "predictions" in explanation:
        pred = explanation["predictions"]
        if isinstance(pred, np.ndarray):
            validation["prediction_shape_match"] = pred.shape == prediction.shape
            validation["prediction_values_match"] = np.allclose(pred, prediction)
    
    validation["is_valid"] = all([
        validation["has_explanation"],
        validation["has_predictions"],
        validation["has_mode"]
    ])
    
    return validation


def aggregate_explanations(
    explanations: List[Dict[str, Any]],
    method: str = "mean"
) -> Dict[str, Any]:
    """
    Aggregate multiple explanations for global analysis.
    
    Args:
        explanations: List of explanation dictionaries
        method: Aggregation method ('mean', 'median', 'max', 'min')
        
    Returns:
        Aggregated explanation dictionary
    """
    if not explanations:
        raise ValueError("Cannot aggregate empty list of explanations")
    
    aggregated = {
        "num_explanations": len(explanations),
        "method": method,
    }
    
    # Aggregate SHAP values if present
    shap_values_list = [
        exp.get("explanations", {}).get("shap", {}).get("shap_values")
        for exp in explanations
        if "explanations" in exp and "shap" in exp["explanations"]
    ]
    
    if shap_values_list:
        shap_array = np.array(shap_values_list)
        if method == "mean":
            aggregated["shap_values"] = np.mean(shap_array, axis=0)
        elif method == "median":
            aggregated["shap_values"] = np.median(shap_array, axis=0)
        elif method == "max":
            aggregated["shap_values"] = np.max(shap_array, axis=0)
        elif method == "min":
            aggregated["shap_values"] = np.min(shap_array, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    # Aggregate LIME feature importance if present
    lime_importance_list = [
        exp.get("explanations", {}).get("lime", {}).get("feature_importance", {})
        for exp in explanations
        if "explanations" in exp and "lime" in exp["explanations"]
    ]
    
    if lime_importance_list:
        # Collect all features
        all_features = set()
        for imp_dict in lime_importance_list:
            if isinstance(imp_dict, dict):
                all_features.update(imp_dict.keys())
        
        # Aggregate importance
        aggregated_importance = {}
        for feature in all_features:
            values = [
                imp_dict.get(feature, 0)
                for imp_dict in lime_importance_list
                if isinstance(imp_dict, dict)
            ]
            if values:
                if method == "mean":
                    aggregated_importance[feature] = np.mean(values)
                elif method == "median":
                    aggregated_importance[feature] = np.median(values)
                elif method == "max":
                    aggregated_importance[feature] = np.max(values)
                elif method == "min":
                    aggregated_importance[feature] = np.min(values)
        
        aggregated["lime_feature_importance"] = aggregated_importance
    
    return aggregated


def format_explanation_for_audit(
    explanation: Dict[str, Any],
    include_raw: bool = False
) -> Dict[str, Any]:
    """
    Format explanation for audit trail and regulatory compliance.
    
    Args:
        explanation: Explanation dictionary
        include_raw: Whether to include raw explanation data
        
    Returns:
        Formatted explanation suitable for audit logs
    """
    audit_record = {
        "timestamp": None,  # Should be set by caller
        "mode": explanation.get("mode", "unknown"),
        "prediction": explanation.get("predictions"),
    }
    
    if include_raw:
        audit_record["raw_explanation"] = explanation
    
    # Format SHAP explanation
    if "explanations" in explanation and "shap" in explanation["explanations"]:
        shap_exp = explanation["explanations"]["shap"]
        audit_record["shap"] = {
            "algorithm": shap_exp.get("algorithm"),
            "feature_importance": shap_exp.get("feature_importance"),
            "has_values": "shap_values" in shap_exp,
        }
    
    # Format LIME explanation
    if "explanations" in explanation and "lime" in explanation["explanations"]:
        lime_exp = explanation["explanations"]["lime"]
        audit_record["lime"] = {
            "num_features": lime_exp.get("num_features"),
            "feature_importance": lime_exp.get("feature_importance"),
            "has_explanations": "explanations" in lime_exp,
        }
    
    return audit_record


def compare_explanations(
    explanation1: Dict[str, Any],
    explanation2: Dict[str, Any],
    metric: str = "cosine"
) -> Dict[str, Any]:
    """
    Compare two explanations to measure consistency.
    
    Args:
        explanation1: First explanation
        explanation2: Second explanation
        metric: Comparison metric ('cosine', 'euclidean', 'manhattan')
        
    Returns:
        Dictionary with comparison results
    """
    comparison = {
        "metric": metric,
    }
    
    # Compare SHAP values if present
    shap1 = explanation1.get("explanations", {}).get("shap", {}).get("shap_values")
    shap2 = explanation2.get("explanations", {}).get("shap", {}).get("shap_values")
    
    if shap1 is not None and shap2 is not None:
        shap1_flat = np.array(shap1).flatten()
        shap2_flat = np.array(shap2).flatten()
        
        if shap1_flat.shape == shap2_flat.shape:
            if metric == "cosine":
                from sklearn.metrics.pairwise import cosine_similarity
                similarity = cosine_similarity(
                    shap1_flat.reshape(1, -1),
                    shap2_flat.reshape(1, -1)
                )[0, 0]
                comparison["shap_similarity"] = float(similarity)
            elif metric == "euclidean":
                distance = np.linalg.norm(shap1_flat - shap2_flat)
                comparison["shap_distance"] = float(distance)
            elif metric == "manhattan":
                distance = np.sum(np.abs(shap1_flat - shap2_flat))
                comparison["shap_distance"] = float(distance)
    
    # Compare predictions
    pred1 = explanation1.get("predictions")
    pred2 = explanation2.get("predictions")
    
    if pred1 is not None and pred2 is not None:
        pred1_flat = np.array(pred1).flatten()
        pred2_flat = np.array(pred2).flatten()
        
        if pred1_flat.shape == pred2_flat.shape:
            comparison["prediction_match"] = np.allclose(pred1_flat, pred2_flat)
            comparison["prediction_difference"] = float(
                np.mean(np.abs(pred1_flat - pred2_flat))
            )
    
    return comparison


def extract_key_features(
    explanation: Dict[str, Any],
    top_k: int = 5,
    explainer_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract top contributing features from an explanation.
    
    Args:
        explanation: Explanation dictionary
        top_k: Number of top features to extract
        explainer_type: Specific explainer to use ('shap' or 'lime')
        
    Returns:
        Dictionary with top features and their contributions
    """
    result = {
        "top_k": top_k,
        "features": []
    }
    
    # Extract from SHAP
    if explainer_type is None or explainer_type == "shap":
        if "explanations" in explanation and "shap" in explanation["explanations"]:
            shap_exp = explanation["explanations"]["shap"]
            shap_values = shap_exp.get("shap_values")
            
            if shap_values is not None:
                shap_array = np.array(shap_values)
                
                # Handle multi-dimensional SHAP values
                if shap_array.ndim > 1:
                    # Average across instances/classes
                    importance = np.abs(shap_array).mean(axis=tuple(range(shap_array.ndim - 1)))
                else:
                    importance = np.abs(shap_array)
                
                # Get top features
                top_indices = np.argsort(importance)[-top_k:][::-1]
                
                feature_names = shap_exp.get("feature_names")
                for idx in top_indices:
                    feature_info = {
                        "index": int(idx),
                        "importance": float(importance[idx]),
                        "shap_value": float(shap_array[..., idx] if shap_array.ndim > 0 else shap_array),
                    }
                    if feature_names and idx < len(feature_names):
                        feature_info["name"] = feature_names[idx]
                    result["features"].append(feature_info)
    
    # Extract from LIME
    if explainer_type is None or explainer_type == "lime":
        if "explanations" in explanation and "lime" in explanation["explanations"]:
            lime_exp = explanation["explanations"]["lime"]
            
            # Handle both single and multiple explanations
            explanations = lime_exp.get("explanations", {})
            if isinstance(explanations, dict):
                explanations = [explanations]
            
            for exp in explanations:
                feature_weights = exp.get("feature_weights", {})
                if feature_weights:
                    sorted_features = sorted(
                        feature_weights.items(),
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )[:top_k]
                    
                    for feature_name, weight in sorted_features:
                        result["features"].append({
                            "name": feature_name,
                            "importance": float(abs(weight)),
                            "weight": float(weight),
                        })
                    break  # Use first explanation
    
    return result



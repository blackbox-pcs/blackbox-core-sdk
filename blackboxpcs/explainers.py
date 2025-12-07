"""
XAI Explainers Module

Implements SHAP and LIME explainers for Black Box Precision framework.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
from .core import ExplanationMode


class BaseExplainer:
    """Base class for all explainers"""
    
    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        **kwargs
    ):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.kwargs = kwargs
    
    def explain(
        self,
        X: np.ndarray,
        mode: ExplanationMode = ExplanationMode.LOCAL,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate explanations for the given input.
        
        Args:
            X: Input data to explain
            mode: Explanation mode (GLOBAL or LOCAL)
            **kwargs: Additional explainer-specific arguments
            
        Returns:
            Dictionary containing explanation results
        """
        raise NotImplementedError("Subclasses must implement explain method")
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Internal prediction method"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'predict'):
            return self.model.predict(X)
        else:
            raise ValueError(
                "Model must have 'predict' or 'predict_proba' method"
            )


class SHAPExplainer(BaseExplainer):
    """
    SHAP (SHapley Additive exPlanations) Explainer.
    
    Provides the theoretical gold standard for feature attribution,
    calculating the fair marginal contribution of each input feature.
    
    Ideal for:
    - Post-mortem auditing and regulatory compliance
    - Understanding feature importance in high-stakes decisions
    - Detecting bias and ensuring non-discriminatory features
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        background_data: Optional[np.ndarray] = None,
        algorithm: str = "auto",
        **kwargs
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model: The black box model to explain
            feature_names: Optional feature names
            class_names: Optional class names for classification
            background_data: Background dataset for SHAP (recommended)
            algorithm: SHAP algorithm ('auto', 'exact', 'permutation', 'sampling', 'tree', 'kernel')
            **kwargs: Additional SHAP parameters
        """
        super().__init__(model, feature_names, class_names, **kwargs)
        self.background_data = background_data
        self.algorithm = algorithm
        self._explainer = None
        self._initialized = False
    
    def _initialize_explainer(self):
        """Lazy initialization of SHAP explainer"""
        if self._initialized:
            return
        
        try:
            import shap
        except ImportError:
            raise ImportError(
                "SHAP is required. Install with: pip install shap"
            )
        
        # Determine model type and select appropriate explainer
        if self.algorithm == "auto":
            # Auto-detect best algorithm
            if hasattr(self.model, 'tree_') or hasattr(self.model, 'estimators_'):
                # Tree-based model
                self.algorithm = "tree"
            elif self.background_data is not None:
                self.algorithm = "kernel"
            else:
                self.algorithm = "permutation"
        
        # Initialize appropriate SHAP explainer
        if self.algorithm == "tree":
            self._explainer = shap.TreeExplainer(self.model, **self.kwargs)
        elif self.algorithm == "kernel":
            if self.background_data is None:
                raise ValueError(
                    "Background data required for KernelExplainer. "
                    "Provide background_data or use algorithm='permutation'"
                )
            self._explainer = shap.KernelExplainer(
                self._predict_wrapper,
                self.background_data,
                **self.kwargs
            )
        elif self.algorithm == "permutation":
            self._explainer = shap.PermutationExplainer(
                self.model,
                self.background_data,
                **self.kwargs
            )
        elif self.algorithm == "exact":
            self._explainer = shap.ExactExplainer(
                self.model,
                self.background_data,
                **self.kwargs
            )
        elif self.algorithm == "sampling":
            self._explainer = shap.SamplingExplainer(
                self.model,
                self.background_data,
                **self.kwargs
            )
        else:
            raise ValueError(f"Unknown SHAP algorithm: {self.algorithm}")
        
        self._initialized = True
    
    def _predict_wrapper(self, X: np.ndarray) -> np.ndarray:
        """Wrapper for model prediction compatible with SHAP"""
        return self._predict(X)
    
    def explain(
        self,
        X: np.ndarray,
        mode: ExplanationMode = ExplanationMode.LOCAL,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations.
        
        Args:
            X: Input data to explain
            mode: Explanation mode (GLOBAL or LOCAL)
            **kwargs: Additional SHAP parameters
            
        Returns:
            Dictionary containing SHAP values and metadata
        """
        self._initialize_explainer()
        
        # Handle single instance
        single_instance = X.ndim == 1
        if single_instance:
            X = X.reshape(1, -1)
        
        # Calculate SHAP values
        shap_values = self._explainer(X, **kwargs)
        
        # Convert to numpy if needed
        if hasattr(shap_values, 'values'):
            shap_array = shap_values.values
            base_values = shap_values.base_values
        else:
            shap_array = np.array(shap_values)
            base_values = None
        
        # Handle single instance output
        if single_instance:
            shap_array = shap_array[0]
            if base_values is not None:
                if isinstance(base_values, np.ndarray) and base_values.ndim > 1:
                    base_values = base_values[0]
        
        # Prepare results
        results = {
            "shap_values": shap_array,
            "base_values": base_values,
            "mode": mode.value,
            "algorithm": self.algorithm,
            "feature_names": self.feature_names,
            "class_names": self.class_names,
        }
        
        # Add feature importance for global mode
        if mode == ExplanationMode.GLOBAL:
            if shap_array.ndim > 1:
                # Multi-class or multi-output
                importance = np.abs(shap_array).mean(axis=0)
            else:
                importance = np.abs(shap_array)
            
            results["feature_importance"] = importance.tolist()
            
            # Create feature importance ranking
            if self.feature_names:
                importance_dict = dict(zip(
                    self.feature_names,
                    importance
                ))
                results["feature_importance_ranking"] = sorted(
                    importance_dict.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )
        
        return results
    
    def get_feature_attribution(
        self,
        X: np.ndarray,
        feature_idx: Optional[int] = None,
        class_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get feature attribution for specific features/classes.
        
        Args:
            X: Input data
            feature_idx: Specific feature index (None for all)
            class_idx: Specific class index for classification (None for all)
            
        Returns:
            Dictionary with feature attributions
        """
        explanation = self.explain(X)
        shap_values = explanation["shap_values"]
        
        if shap_values.ndim > 1:
            if class_idx is not None:
                shap_values = shap_values[:, class_idx]
            elif shap_values.shape[1] == 1:
                shap_values = shap_values[:, 0]
        
        if feature_idx is not None:
            attributions = shap_values[..., feature_idx]
        else:
            attributions = shap_values
        
        result = {
            "attributions": attributions,
            "feature_idx": feature_idx,
            "class_idx": class_idx,
        }
        
        if self.feature_names and feature_idx is not None:
            result["feature_name"] = self.feature_names[feature_idx]
        
        if self.class_names and class_idx is not None:
            result["class_name"] = self.class_names[class_idx]
        
        return result


class LIMEExplainer(BaseExplainer):
    """
    LIME (Local Interpretable Model-agnostic Explanations) Explainer.
    
    Provides fast, intuitive explanations by training a simple, local
    surrogate model around a single prediction point.
    
    Ideal for:
    - Real-time operational oversight
    - Split-second decision validation
    - Quick feature identification for critical actions
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        mode: str = "classification",
        num_features: int = 10,
        **kwargs
    ):
        """
        Initialize LIME explainer.
        
        Args:
            model: The black box model to explain
            feature_names: Optional feature names
            class_names: Optional class names for classification
            mode: Task mode ('classification' or 'regression')
            num_features: Number of top features to show in explanation
            **kwargs: Additional LIME parameters
        """
        super().__init__(model, feature_names, class_names, **kwargs)
        self.mode = mode
        self.num_features = num_features
        self._explainer = None
        self._initialized = False
    
    def _initialize_explainer(self, X_sample: np.ndarray):
        """Lazy initialization of LIME explainer"""
        if self._initialized:
            return
        
        try:
            import lime
            from lime import lime_tabular
        except ImportError:
            raise ImportError(
                "LIME is required. Install with: pip install lime"
            )
        
        # Determine training data shape for initialization
        if X_sample.ndim == 1:
            n_features = X_sample.shape[0]
        else:
            n_features = X_sample.shape[1]
        
        # Initialize LIME explainer
        self._explainer = lime_tabular.LimeTabularExplainer(
            training_data=None,  # Will use model directly
            mode=self.mode,
            feature_names=self.feature_names,
            class_names=self.class_names,
            num_features=n_features,
            **self.kwargs
        )
        
        self._initialized = True
    
    def explain(
        self,
        X: np.ndarray,
        mode: ExplanationMode = ExplanationMode.LOCAL,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate LIME explanations.
        
        Args:
            X: Input data to explain
            mode: Explanation mode (GLOBAL or LOCAL)
            **kwargs: Additional LIME parameters
            
        Returns:
            Dictionary containing LIME explanations
        """
        # Handle single instance
        single_instance = X.ndim == 1
        if single_instance:
            X = X.reshape(1, -1)
        
        num_features = kwargs.get("num_features", self.num_features)
        explanations = []
        
        for instance in X:
            self._initialize_explainer(instance)
            
            # Generate LIME explanation
            explanation = self._explainer.explain_instance(
                instance,
                self._predict,
                num_features=num_features,
                **{k: v for k, v in kwargs.items() if k != "num_features"}
            )
            
            # Extract explanation data
            exp_dict = {
                "feature_weights": dict(explanation.as_list()),
                "score": explanation.score,
                "intercept": explanation.intercept[0] if hasattr(explanation, 'intercept') else None,
            }
            
            # Get feature indices and values
            if hasattr(explanation, 'as_map'):
                exp_map = explanation.as_map()
                exp_dict["feature_map"] = exp_map
            
            explanations.append(exp_dict)
        
        # Prepare results
        results = {
            "explanations": explanations if not single_instance else explanations[0],
            "mode": mode.value,
            "num_features": num_features,
            "feature_names": self.feature_names,
            "class_names": self.class_names,
        }
        
        # For global mode, aggregate feature importance
        if mode == ExplanationMode.GLOBAL and not single_instance:
            all_weights = {}
            for exp in explanations:
                for feature, weight in exp["feature_weights"].items():
                    if feature not in all_weights:
                        all_weights[feature] = []
                    all_weights[feature].append(abs(weight))
            
            # Average absolute importance
            feature_importance = {
                feature: np.mean(weights)
                for feature, weights in all_weights.items()
            }
            
            results["feature_importance"] = feature_importance
            results["feature_importance_ranking"] = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
        
        return results
    
    def get_top_features(
        self,
        X: np.ndarray,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get top contributing features for the prediction.
        
        Args:
            X: Input data (single instance)
            top_k: Number of top features to return (None for all)
            
        Returns:
            Dictionary with top features and their contributions
        """
        explanation = self.explain(X)
        
        if isinstance(explanation["explanations"], list):
            exp = explanation["explanations"][0]
        else:
            exp = explanation["explanations"]
        
        feature_weights = exp["feature_weights"]
        
        # Sort by absolute weight
        sorted_features = sorted(
            feature_weights.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        if top_k is not None:
            sorted_features = sorted_features[:top_k]
        
        return {
            "top_features": sorted_features,
            "num_features": len(sorted_features),
            "total_features": len(feature_weights),
        }



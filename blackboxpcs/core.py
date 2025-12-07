"""
Core Black Box Precision Framework

The main framework class that integrates XAI techniques with black box models
to ensure transparency, safety, and accountability without sacrificing performance.
"""

from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
from enum import Enum


class ExplanationType(Enum):
    """Types of explanations supported by the framework"""
    SHAP = "shap"
    LIME = "lime"
    BOTH = "both"


class ExplanationMode(Enum):
    """Modes of explanation: global (auditing) or local (operational)"""
    GLOBAL = "global"  # For auditing and development
    LOCAL = "local"    # For operational oversight


class BlackBoxPrecision:
    """
    Core framework for Black Box Precision.
    
    Integrates XAI techniques (SHAP and LIME) with black box models to provide
    explainable AI capabilities for high-stakes applications.
    
    Attributes:
        model: The black box model (must support predict/predict_proba)
        explainer_type: Type of explainer to use (SHAP, LIME, or BOTH)
        feature_names: Optional list of feature names for better explanations
        class_names: Optional list of class names for classification tasks
    """
    
    def __init__(
        self,
        model: Any,
        explainer_type: ExplanationType = ExplanationType.BOTH,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize Black Box Precision framework.
        
        Args:
            model: The black box model to explain
            explainer_type: Type of explainer(s) to use
            feature_names: Optional feature names for interpretability
            class_names: Optional class names for classification
            **kwargs: Additional arguments passed to explainers
        """
        self.model = model
        self.explainer_type = explainer_type
        self.feature_names = feature_names
        self.class_names = class_names
        self.kwargs = kwargs
        
        # Lazy initialization of explainers
        self._shap_explainer = None
        self._lime_explainer = None
        
    def _get_shap_explainer(self, X_background: Optional[np.ndarray] = None):
        """Lazy initialization of SHAP explainer"""
        if self._shap_explainer is None:
            from .explainers import SHAPExplainer
            self._shap_explainer = SHAPExplainer(
                self.model,
                feature_names=self.feature_names,
                class_names=self.class_names,
                background_data=X_background,
                **self.kwargs
            )
        return self._shap_explainer
    
    def _get_lime_explainer(self, X_background: Optional[np.ndarray] = None):
        """Lazy initialization of LIME explainer"""
        if self._lime_explainer is None:
            from .explainers import LIMEExplainer
            self._lime_explainer = LIMEExplainer(
                self.model,
                feature_names=self.feature_names,
                class_names=self.class_names,
                **self.kwargs
            )
        return self._lime_explainer
    
    def explain(
        self,
        X: np.ndarray,
        mode: ExplanationMode = ExplanationMode.LOCAL,
        explanation_type: Optional[ExplanationType] = None,
        X_background: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate explanations for predictions.
        
        Args:
            X: Input data to explain (single instance or batch)
            mode: Explanation mode (GLOBAL for auditing, LOCAL for operational)
            explanation_type: Override default explainer type
            X_background: Background data for SHAP (optional)
            **kwargs: Additional arguments for explainers
            
        Returns:
            Dictionary containing explanations and metadata
        """
        explanation_type = explanation_type or self.explainer_type
        
        results = {
            "predictions": self._predict(X),
            "mode": mode.value,
            "explanations": {}
        }
        
        # Generate SHAP explanations if requested
        if explanation_type in [ExplanationType.SHAP, ExplanationType.BOTH]:
            shap_explainer = self._get_shap_explainer(X_background)
            results["explanations"]["shap"] = shap_explainer.explain(
                X, mode=mode, **kwargs
            )
        
        # Generate LIME explanations if requested
        if explanation_type in [ExplanationType.LIME, ExplanationType.BOTH]:
            lime_explainer = self._get_lime_explainer(X_background)
            results["explanations"]["lime"] = lime_explainer.explain(
                X, mode=mode, **kwargs
            )
        
        return results
    
    def explain_local(
        self,
        X: np.ndarray,
        explanation_type: Optional[ExplanationType] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate local explanations for operational oversight.
        
        Convenience method for real-time explanation generation.
        
        Args:
            X: Input data to explain
            explanation_type: Override default explainer type
            **kwargs: Additional arguments for explainers
            
        Returns:
            Dictionary containing local explanations
        """
        return self.explain(
            X,
            mode=ExplanationMode.LOCAL,
            explanation_type=explanation_type,
            **kwargs
        )
    
    def explain_global(
        self,
        X: np.ndarray,
        explanation_type: Optional[ExplanationType] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate global explanations for auditing and development.
        
        Convenience method for detecting systemic biases and model behavior.
        
        Args:
            X: Input data to explain (typically training/validation set)
            explanation_type: Override default explainer type
            **kwargs: Additional arguments for explainers
            
        Returns:
            Dictionary containing global explanations
        """
        return self.explain(
            X,
            mode=ExplanationMode.GLOBAL,
            explanation_type=explanation_type,
            **kwargs
        )
    
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
    
    def predict_with_explanation(
        self,
        X: np.ndarray,
        explanation_type: Optional[ExplanationType] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make predictions with immediate explanations.
        
        This is the primary method for operational use, providing both
        the prediction and its explanation in a single call.
        
        Args:
            X: Input data
            explanation_type: Override default explainer type
            **kwargs: Additional arguments for explainers
            
        Returns:
            Dictionary with predictions and explanations
        """
        return self.explain_local(X, explanation_type=explanation_type, **kwargs)
    
    def audit_model(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        explanation_type: Optional[ExplanationType] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform comprehensive model auditing using global XAI.
        
        Detects systemic biases and validates model behavior across the dataset.
        
        Args:
            X: Dataset for auditing (typically training/validation set)
            y: Optional ground truth labels for validation
            explanation_type: Override default explainer type
            **kwargs: Additional arguments for explainers
            
        Returns:
            Dictionary containing audit results and global explanations
        """
        results = self.explain_global(X, explanation_type=explanation_type, **kwargs)
        
        if y is not None:
            predictions = self._predict(X)
            if predictions.ndim > 1:
                predictions = np.argmax(predictions, axis=1)
            results["accuracy"] = float(np.mean(predictions == y))
            results["ground_truth_provided"] = True
        else:
            results["ground_truth_provided"] = False
        
        return results



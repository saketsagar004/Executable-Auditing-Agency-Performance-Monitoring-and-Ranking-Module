"""Support Vector Machine Classifier implementation wrapping scikit-learn SVC."""

from typing import Any, Dict, Optional
import numpy as np
from sklearn.svm import SVC


class SVMModel:
    """Support Vector Machine Classifier with calibrated probabilities."""

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: str = "scale",
        probability: bool = True,
        random_state: int = 20260824,
    ):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.probability = probability
        self.random_state = random_state

        self.model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=probability,
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMModel":
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predict.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predict_proba.")
        return self.model.predict_proba(X)

    def get_params(self) -> Dict[str, Any]:
        return {
            "C": self.C,
            "kernel": self.kernel,
            "gamma": self.gamma,
            "probability": self.probability,
            "random_state": self.random_state,
        }

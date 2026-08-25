"""Gradient Boosting Classifier implementation wrapping scikit-learn GradientBoostingClassifier."""

from typing import Any, Dict, Optional
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier


class GradientBoostingModel:
    """Gradient Boosted Decision Trees classifier for performance renewal prediction."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        subsample: float = 0.8,
        random_state: int = 20260824,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state

        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingModel":
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
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "random_state": self.random_state,
        }

    @property
    def feature_importances_(self) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted to access feature importances.")
        return self.model.feature_importances_

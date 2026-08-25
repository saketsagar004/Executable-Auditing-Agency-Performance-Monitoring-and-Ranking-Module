"""Baseline Classifier implementation wrapping scikit-learn DummyClassifier."""

from typing import Any, Dict, Optional
import numpy as np
from sklearn.dummy import DummyClassifier


class BaselineModel:
    """Simple baseline classifier implementing prior or majority-class prediction."""

    def __init__(self, strategy: str = "most_frequent", random_state: int = 20260824):
        self.strategy = strategy
        self.random_state = random_state
        self.model = DummyClassifier(strategy=strategy, random_state=random_state)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineModel":
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
        return {"strategy": self.strategy, "random_state": self.random_state}

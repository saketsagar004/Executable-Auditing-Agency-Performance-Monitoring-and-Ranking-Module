"""Multi-Layer Perceptron (MLP) Neural Network wrapper for renewal prediction."""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.neural_network import MLPClassifier


class NeuralNetworkModel:
    """Multi-Layer Perceptron Neural Network classifier."""

    def __init__(
        self,
        hidden_layer_sizes: Any = (32, 16),
        activation: str = "relu",
        solver: str = "adam",
        alpha: float = 0.001,
        learning_rate_init: float = 0.001,
        max_iter: int = 500,
        early_stopping: bool = True,
        random_state: int = 20260824,
    ):
        if isinstance(hidden_layer_sizes, list):
            hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.random_state = random_state

        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralNetworkModel":
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
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
            "activation": self.activation,
            "solver": self.solver,
            "alpha": self.alpha,
            "learning_rate_init": self.learning_rate_init,
            "max_iter": self.max_iter,
            "early_stopping": self.early_stopping,
            "random_state": self.random_state,
        }

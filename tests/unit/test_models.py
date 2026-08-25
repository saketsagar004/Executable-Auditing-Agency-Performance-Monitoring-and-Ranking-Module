"""Unit tests for classification models, cross-validation, and evaluation metrics."""

import json
import os
import unittest
import numpy as np

from src.evaluation.metrics import ModelEvaluator
from src.features.extractor import FeatureExtractor
from src.models.baseline import BaselineModel
from src.models.gradient_boosting import GradientBoostingModel
from src.models.model_trainer import ModelPipelineTrainer
from src.models.neural_network import NeuralNetworkModel
from src.models.svm import SVMModel


class TestClassificationModels(unittest.TestCase):
    """Verifies all four required classification models and evaluation metrics."""

    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        cls.dataset_path = os.path.join(cls.base_dir, "data", "sample_dataset.json")
        cls.model_config_path = os.path.join(cls.base_dir, "config", "model_config.json")

        # Create dummy synthetic data matrix for unit testing
        np.random.seed(42)
        cls.X = np.random.randn(50, 12)
        cls.y = np.random.choice([0, 1], size=50, p=[0.3, 0.7])

    def test_baseline_model(self):
        model = BaselineModel(strategy="most_frequent")
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), 50)
        self.assertTrue(all(p in [0, 1] for p in preds))

    def test_gradient_boosting_model(self):
        model = GradientBoostingModel(n_estimators=10, max_depth=2, random_state=42)
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        probs = model.predict_proba(self.X)
        self.assertEqual(len(preds), 50)
        self.assertEqual(probs.shape, (50, 2))
        self.assertEqual(len(model.feature_importances_), 12)

    def test_svm_model(self):
        model = SVMModel(C=1.0, kernel="rbf", probability=True, random_state=42)
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        probs = model.predict_proba(self.X)
        self.assertEqual(len(preds), 50)
        self.assertEqual(probs.shape, (50, 2))

    def test_neural_network_model(self):
        model = NeuralNetworkModel(hidden_layer_sizes=(16, 8), max_iter=100, random_state=42)
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        probs = model.predict_proba(self.X)
        self.assertEqual(len(preds), 50)
        self.assertEqual(probs.shape, (50, 2))

    def test_evaluator_metrics_calculation(self):
        y_true = np.array([1, 1, 1, 0, 0, 1, 0, 1])
        y_pred = np.array([1, 1, 0, 0, 0, 1, 1, 1])
        metrics = ModelEvaluator.evaluate_predictions(y_true, y_pred)

        self.assertIn("macro_f1", metrics)
        self.assertIn("accuracy", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertIn("per_class_results", metrics)
        self.assertEqual(metrics["primary_metric"], "macro_f1")
        self.assertGreaterEqual(metrics["macro_f1"], 0.0)
        self.assertLessEqual(metrics["macro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()

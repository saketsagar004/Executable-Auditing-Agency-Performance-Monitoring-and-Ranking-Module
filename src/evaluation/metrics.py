"""Evaluation harness computing standard classification metrics, Macro-F1, confusion matrices, and per-class breakdowns."""

import json
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


class ModelEvaluator:
    """Evaluates classification models and compiles structured JSON reports."""

    CLASS_NAMES = ["NOT_RENEWED", "RENEWED"]

    @classmethod
    def evaluate_predictions(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Calculates comprehensive evaluation metrics."""
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)

        acc = float(accuracy_score(y_true, y_pred))
        prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        prec_binary = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        rec_binary = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        f1_binary = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]

        # Per-class detailed breakdown
        prec_per, rec_per, f1_per, sup_per = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1], zero_division=0
        )

        per_class = {
            "NOT_RENEWED (Class 0)": {
                "precision": round(float(prec_per[0]), 4),
                "recall": round(float(rec_per[0]), 4),
                "f1_score": round(float(f1_per[0]), 4),
                "support": int(sup_per[0]),
            },
            "RENEWED (Class 1)": {
                "precision": round(float(prec_per[1]), 4),
                "recall": round(float(rec_per[1]), 4),
                "f1_score": round(float(f1_per[1]), 4),
                "support": int(sup_per[1]),
            },
        }

        return {
            "primary_metric": "macro_f1",
            "macro_f1": round(f1_macro, 4),
            "accuracy": round(acc, 4),
            "macro_precision": round(prec_macro, 4),
            "macro_recall": round(rec_macro, 4),
            "binary_precision": round(prec_binary, 4),
            "binary_recall": round(rec_binary, 4),
            "binary_f1": round(f1_binary, 4),
            "confusion_matrix": {
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
                "matrix": cm.tolist(),
            },
            "per_class_results": per_class,
            "sample_count": len(y_true),
        }

"""Métriques d'évaluation d'un classifieur binaire de départ client."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray | pd.Series, y_proba: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    """Calcule les métriques à partir des probabilités de départ (classe 1).

    - ROC-AUC et PR-AUC dépendent seulement du classement des probabilités (pas du seuil).
    - Précision, rappel, F1, exactitude et matrice de confusion dépendent du seuil.
    La PR-AUC est estimée par l'average precision de scikit-learn.
    """
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def metrics_table(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Met les métriques de plusieurs modèles côte à côte (une ligne par modèle)."""
    rows = {
        name: {
            **{k: v for k, v in m.items() if k != "confusion_matrix"},
            **m["confusion_matrix"],
        }
        for name, m in results.items()
    }
    table = pd.DataFrame(rows).T
    counts = ["tn", "fp", "fn", "tp"]
    return table.astype(
        {c: float for c in table.columns if c not in counts} | dict.fromkeys(counts, int)
    )

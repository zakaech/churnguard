"""Chargement du modèle sauvegardé et prédiction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from churnguard.config import load_config, resolve_path


def load_model(path: str | Path | None = None) -> dict[str, Any]:
    """Charge l'artefact unique : pipeline, seuil, version et listes de variables."""
    if path is None:
        path = resolve_path(load_config()["model"]["artifact"])
    return joblib.load(path)


def predict_proba(artifact: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    """Probabilité de départ (classe 1) pour chaque client."""
    features = artifact["numeric_features"] + artifact["categorical_features"]
    return artifact["pipeline"].predict_proba(X[features])[:, 1]


def predict(artifact: dict[str, Any], X: pd.DataFrame) -> pd.DataFrame:
    """Probabilité et décision (probabilité >= seuil du modèle) pour chaque client."""
    proba = predict_proba(artifact, X)
    return pd.DataFrame(
        {"churn_probability": proba, "churn_predicted": proba >= artifact["threshold"]},
        index=X.index,
    )

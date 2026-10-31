"""Chargement du modèle sauvegardé, prédiction et niveau de risque."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from churnguard.config import load_config, resolve_path

RISK_LEVELS = ("faible", "moyen", "élevé")


def load_model(path: str | Path | None = None) -> dict[str, Any]:
    """Charge l'artefact unique : pipeline, seuil, version et listes de variables."""
    if path is None:
        path = resolve_path(load_config()["model"]["artifact"])
    return joblib.load(path)


def predict_proba(artifact: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    """Probabilité de départ (classe 1) pour chaque client."""
    features = artifact["numeric_features"] + artifact["categorical_features"]
    return artifact["pipeline"].predict_proba(X[features])[:, 1]


def check_risk_bounds(threshold: float, medium_from: float) -> None:
    """Vérifie que la limite faible/moyen est strictement entre 0 et le seuil de décision."""
    if not 0 < medium_from < threshold:
        raise ValueError(
            f"Limite faible/moyen ({medium_from}) incohérente avec le seuil ({threshold}) : "
            "elle doit être strictement comprise entre 0 et le seuil."
        )


def risk_level(proba: np.ndarray, threshold: float, medium_from: float) -> np.ndarray:
    """Niveau de risque : « élevé » exactement quand le client est signalé (proba >= seuil).

    Sous le seuil : « moyen » si proba >= medium_from, sinon « faible ».
    """
    check_risk_bounds(threshold, medium_from)
    proba = np.asarray(proba)
    return np.select(
        [proba >= threshold, proba >= medium_from], ["élevé", "moyen"], default="faible"
    )


def predict(
    artifact: dict[str, Any], X: pd.DataFrame, medium_from: float | None = None
) -> pd.DataFrame:
    """Probabilité, décision (probabilité >= seuil du modèle) et niveau de risque."""
    if medium_from is None:
        medium_from = load_config()["risk_levels"]["medium_from"]
    proba = predict_proba(artifact, X)
    threshold = artifact["threshold"]
    return pd.DataFrame(
        {
            "churn_probability": proba,
            "churn_predicted": proba >= threshold,
            "risk_level": risk_level(proba, threshold, medium_from),
        },
        index=X.index,
    )

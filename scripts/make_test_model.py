"""Entraîne un petit modèle de TEST sur l'échantillon synthétique (tests/data/sample.csv).

Utilisé par la CI pour lancer l'API en conteneur sans Kaggle ni vrai modèle. Il n'utilise
que les dépendances de l'image de l'API (pas de mlflow ni de xgboost). Ses prédictions n'ont
aucune valeur métier.

Usage : PYTHONPATH=src python scripts/make_test_model.py --out ci-model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier

from churnguard.config import PROJECT_ROOT, load_config
from churnguard.data import clean, split_data, split_xy
from churnguard.features import build_preprocessor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("ci-model"))
    args = parser.parse_args()

    cfg = load_config()
    raw = pd.read_csv(PROJECT_ROOT / "tests" / "data" / "sample.csv")
    X, y = split_xy(split_data(clean(raw, cfg), cfg)["train"], cfg)
    pipeline = ImbPipeline(
        [
            ("preprocess", build_preprocessor(cfg)),
            ("model", RandomForestClassifier(n_estimators=50, random_state=cfg["seed"])),
        ]
    ).fit(X, y)

    args.out.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "threshold": 0.3,
        "model_name": "random_forest",
        "strategy": "none",
        "version": "ci-test",
        "numeric_features": cfg["columns"]["numeric"],
        "categorical_features": cfg["columns"]["categorical"],
    }
    joblib.dump(artifact, args.out / "churnguard_model.joblib")
    metadata = {
        "model": "random_forest",
        "version": "ci-test",
        "threshold": 0.3,
        "threshold_rule": "modèle de test (échantillon synthétique), sans valeur métier",
        "validation_metrics": None,
        "test_metrics": None,
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Modèle de test écrit dans {args.out}/")


if __name__ == "__main__":
    main()

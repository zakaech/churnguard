"""Entraînement des modèles de référence (baselines) et évaluation sur la validation."""

from __future__ import annotations

import json
from typing import Any

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churnguard.config import load_config, resolve_path
from churnguard.data import get_splits, split_xy
from churnguard.evaluate import compute_metrics, metrics_table
from churnguard.features import build_preprocessor


def build_baselines(cfg: dict[str, Any]) -> dict[str, Pipeline]:
    """Deux points de comparaison, dans des pipelines scikit-learn.

    - dummy_most_frequent : prédit toujours la classe majoritaire (« resté »).
      C'est le plancher : tout modèle utile doit faire mieux.
    - logistic_regression : modèle linéaire simple, sans pondération des classes
      (le déséquilibre sera traité en phase 2).
    """
    seed = cfg["seed"]
    return {
        "dummy_most_frequent": Pipeline([("model", DummyClassifier(strategy="most_frequent"))]),
        "logistic_regression": Pipeline(
            [
                ("preprocess", build_preprocessor(cfg)),
                ("model", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        ),
    }


def train_baselines(
    cfg: dict[str, Any] | None = None, threshold: float = 0.5
) -> dict[str, dict[str, Any]]:
    """Entraîne les baselines sur le train et renvoie leurs métriques sur la validation.

    Le seuil reste à 0,5 ici : il ne sert qu'à comparer les baselines. Le choix du seuil
    selon l'objectif métier se fera en phase 2, sur la validation.
    """
    cfg = cfg or load_config()
    splits = get_splits(cfg)
    X_train, y_train = split_xy(splits["train"], cfg)
    X_val, y_val = split_xy(splits["val"], cfg)

    results = {}
    for name, pipeline in build_baselines(cfg).items():
        pipeline.fit(X_train, y_train)
        y_proba = pipeline.predict_proba(X_val)[:, 1]
        results[name] = compute_metrics(y_val, y_proba, threshold=threshold)
    return results


def main() -> None:
    cfg = load_config()
    results = train_baselines(cfg)

    table = metrics_table(results)
    print("Baselines — métriques sur la VALIDATION (seuil 0,5) :\n")
    print(table.to_string(float_format=lambda x: f"{x:.3f}"))

    out = resolve_path(cfg["paths"]["reports_dir"]) / "baseline_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nMétriques enregistrées dans {out}")


if __name__ == "__main__":
    main()

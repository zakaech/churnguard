"""Entraînement : baselines, comparaison des modèles, recherche d'hyperparamètres,
choix du modèle final et de son seuil sur la validation, sauvegarde.

Toutes les expériences sont faites sur le train (validation croisée) puis la validation ;
le test n'est jamais utilisé ici (voir `evaluate_on_test` dans `churnguard.evaluate`).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churnguard.config import PROJECT_ROOT, load_config, resolve_path
from churnguard.data import get_splits, split_xy
from churnguard.evaluate import (
    compute_metrics,
    cost_analysis,
    cost_curve,
    cost_parameters,
    metrics_table,
    plot_cost_curve,
    plot_threshold_analysis,
    precision_at_recall,
    select_threshold,
)
from churnguard.features import build_preprocessor

MODELS = ("logistic_regression", "random_forest", "xgboost")
STRATEGIES = ("none", "class_weight", "smotenc")

# Plages de recherche volontairement raisonnables (préfixe "model__" : étape du pipeline).
SEARCH_SPACES: dict[str, dict[str, Any]] = {
    "logistic_regression": {"model__C": loguniform(1e-3, 1e2)},
    "random_forest": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 4, 6, 8, 12, 16],
        "model__min_samples_leaf": randint(1, 30),
        "model__max_features": ["sqrt", 0.3, 0.5],
    },
    "xgboost": {
        "model__n_estimators": randint(100, 600),
        "model__learning_rate": loguniform(0.01, 0.3),
        "model__max_depth": randint(2, 7),
        "model__min_child_weight": randint(1, 10),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.5, 0.5),
        "model__reg_lambda": loguniform(0.1, 10),
    },
}


# --- Construction des modèles -------------------------------------------------------------


def build_baselines(cfg: dict[str, Any]) -> dict[str, Pipeline]:
    """Deux points de comparaison, dans des pipelines scikit-learn.

    - dummy_most_frequent : prédit toujours la classe majoritaire (« resté »).
      C'est le plancher : tout modèle utile doit faire mieux.
    - logistic_regression : modèle linéaire simple, sans pondération des classes.
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


def build_estimator(name: str, strategy: str, seed: int, pos_weight: float) -> Any:
    """Classifieur avec ses réglages par défaut et la pondération des classes si demandée.

    `pos_weight` (négatifs / positifs du train) sert à XGBoost, qui n'a pas de
    `class_weight` : il multiplie le poids des exemples positifs.
    """
    weighted = strategy == "class_weight"
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=2000, class_weight="balanced" if weighted else None, random_state=seed
        )
    if name == "random_forest":
        return RandomForestClassifier(
            class_weight="balanced" if weighted else None, n_jobs=-1, random_state=seed
        )
    if name == "xgboost":
        return XGBClassifier(
            scale_pos_weight=pos_weight if weighted else 1.0,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"Modèle inconnu : {name}")


def build_pipeline(
    name: str, strategy: str, cfg: dict[str, Any], y_train: pd.Series
) -> ImbPipeline:
    """Pipeline complet : [SMOTENC] -> prétraitement -> modèle.

    Le pipeline d'imbalanced-learn applique le rééchantillonnage uniquement pendant `fit`
    (donc sur les plis d'entraînement) et l'ignore pendant `predict` : les données évaluées
    ne sont jamais suréchantillonnées. SMOTENC est placé avant l'encodage car il gère
    lui-même les variables catégorielles (il ne crée pas de « 0,37 contrat mensuel »).
    """
    seed = cfg["seed"]
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    steps: list[tuple[str, Any]] = []
    if strategy == "smotenc":
        steps.append(
            (
                "smotenc",
                SMOTENC(categorical_features=cfg["columns"]["categorical"], random_state=seed),
            )
        )
    elif strategy not in STRATEGIES:
        raise ValueError(f"Stratégie inconnue : {strategy}")
    steps += [
        ("preprocess", build_preprocessor(cfg)),
        ("model", build_estimator(name, strategy, seed, pos_weight)),
    ]
    return ImbPipeline(steps)


# --- Validation croisée et recherche ------------------------------------------------------


def make_scoring(target_recall: float) -> dict[str, Any]:
    """Métriques calculées à chaque pli. La PR-AUC sert à sélectionner (refit)."""
    return {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
        "precision_at_recall": make_scorer(
            precision_at_recall, response_method="predict_proba", target_recall=target_recall
        ),
    }


def make_cv(cfg: dict[str, Any]) -> StratifiedKFold:
    return StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=cfg["seed"])


def cross_validate_pipeline(
    pipeline: ImbPipeline, X: pd.DataFrame, y: pd.Series, cfg: dict[str, Any]
) -> dict[str, float]:
    """Validation croisée stratifiée sur le train : moyenne et écart-type de chaque métrique."""
    scores = cross_validate(
        pipeline, X, y, cv=make_cv(cfg), scoring=make_scoring(cfg["threshold"]["target_recall"])
    )
    out = {}
    for metric in ("pr_auc", "roc_auc", "precision_at_recall"):
        values = scores[f"test_{metric}"]
        out[f"cv_{metric}_mean"] = float(np.mean(values))
        out[f"cv_{metric}_std"] = float(np.std(values))
    return out


def search_hyperparameters(
    name: str, strategy: str, X: pd.DataFrame, y: pd.Series, cfg: dict[str, Any]
) -> RandomizedSearchCV:
    """Recherche aléatoire d'hyperparamètres, sélection sur la PR-AUC moyenne en CV."""
    search = RandomizedSearchCV(
        build_pipeline(name, strategy, cfg, y),
        SEARCH_SPACES[name],
        n_iter=cfg["search"]["n_iter"],
        scoring=make_scoring(cfg["threshold"]["target_recall"]),
        refit="pr_auc",
        cv=make_cv(cfg),
        random_state=cfg["seed"],
    )
    return search.fit(X, y)


def search_cv_metrics(search: RandomizedSearchCV) -> dict[str, float]:
    """Métriques de CV (moyenne, écart-type) de la meilleure combinaison trouvée."""
    i = search.best_index_
    res = search.cv_results_
    return {
        f"cv_{m}_{stat}": float(res[f"{stat}_test_{m}"][i])
        for m in ("pr_auc", "roc_auc", "precision_at_recall")
        for stat in ("mean", "std")
    }


# --- MLflow --------------------------------------------------------------------------------


def setup_mlflow(cfg: dict[str, Any]) -> None:
    """Suivi local : base SQLite + dossier d'artefacts, à la racine du projet."""
    db = resolve_path(cfg["mlflow"]["db"]).as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    name = cfg["mlflow"]["experiment"]
    if mlflow.get_experiment_by_name(name) is None:
        artifacts = resolve_path(cfg["mlflow"]["artifacts_dir"]).as_uri()
        mlflow.create_experiment(name, artifact_location=artifacts)
    mlflow.set_experiment(name)


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    return {k.removeprefix("model__"): v for k, v in params.items()}


def _flat_val_metrics(m: dict[str, Any], prefix: str = "val") -> dict[str, float]:
    flat = {f"{prefix}_{k}": float(v) for k, v in m.items() if k != "confusion_matrix"}
    flat.update({f"{prefix}_{k}": float(v) for k, v in m["confusion_matrix"].items()})
    return flat


# --- Chaîne complète -----------------------------------------------------------------------


def train_baselines(
    cfg: dict[str, Any] | None = None, threshold: float = 0.5
) -> dict[str, dict[str, Any]]:
    """Entraîne les baselines sur le train et renvoie leurs métriques sur la validation."""
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


def check_final_model_not_tested(cfg: dict[str, Any], force: bool = False) -> None:
    """Refuse de réentraîner si le modèle final a déjà été évalué sur le test.

    Un nouvel entraînement remplacerait le modèle et son seuil : les métriques de test
    existantes ne correspondraient plus au modèle sauvegardé.
    """
    test_report = resolve_path(cfg["paths"]["reports_dir"]) / "test_metrics.json"
    if not test_report.exists():
        return
    message = (
        f"Le modèle final a déjà été évalué sur le test ({test_report}). "
        "Un nouvel entraînement remplacerait le modèle et son seuil : ces métriques de test "
        "deviendraient obsolètes, et le test ne pourrait plus être réévalué honnêtement."
    )
    if not force:
        raise RuntimeError(message + " Utiliser --force seulement en connaissance de cause.")
    print(f"ATTENTION (--force) : {message}", flush=True)


def run_experiments(cfg: dict[str, Any] | None = None, force: bool = False) -> dict[str, Any]:
    """Étapes A à D : comparaison en CV, recherche, choix sur la validation, sauvegarde."""
    cfg = cfg or load_config()
    check_final_model_not_tested(cfg, force=force)
    target_recall = cfg["threshold"]["target_recall"]
    splits = get_splits(cfg)
    X_train, y_train = split_xy(splits["train"], cfg)
    X_val, y_val = split_xy(splits["val"], cfg)
    setup_mlflow(cfg)
    reports = resolve_path(cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)

    # Étape A : 3 modèles × 3 stratégies de déséquilibre, en validation croisée sur le train.
    rows = []
    for name in MODELS:
        for strategy in STRATEGIES:
            print(f"[CV] {name} / {strategy} ...", flush=True)
            pipeline = build_pipeline(name, strategy, cfg, y_train)
            scores = cross_validate_pipeline(pipeline, X_train, y_train, cfg)
            with mlflow.start_run(run_name=f"cv_{name}_{strategy}"):
                mlflow.set_tags({"stage": "cv", "model": name, "strategy": strategy})
                mlflow.log_params({"model": name, "strategy": strategy, "params": "défaut"})
                mlflow.log_metrics(scores)
            rows.append({"stage": "A_cv_defaut", "model": name, "strategy": strategy, **scores})
    stage_a = pd.DataFrame(rows)
    best_strategy = (
        stage_a.loc[stage_a.groupby("model")["cv_pr_auc_mean"].idxmax()]
        .set_index("model")["strategy"]
        .to_dict()
    )

    # Étape B : recherche d'hyperparamètres avec la meilleure stratégie de chaque modèle.
    # On garde la version réglée seulement si elle bat la version par défaut en CV.
    candidates: dict[str, dict[str, Any]] = {}
    for name in MODELS:
        strategy = best_strategy[name]
        print(f"[Recherche] {name} / {strategy} ...", flush=True)
        search = search_hyperparameters(name, strategy, X_train, y_train, cfg)
        scores = search_cv_metrics(search)
        params = _clean_params(search.best_params_)
        with mlflow.start_run(run_name=f"search_{name}_{strategy}"):
            mlflow.set_tags({"stage": "search", "model": name, "strategy": strategy})
            mlflow.log_params({"model": name, "strategy": strategy, **params})
            mlflow.log_metrics(scores)
        rows.append({"stage": "B_recherche", "model": name, "strategy": strategy, **scores})

        default = stage_a.query("model == @name and strategy == @strategy").iloc[0]
        if scores["cv_pr_auc_mean"] > default["cv_pr_auc_mean"]:
            candidates[name] = {
                "strategy": strategy, "tuned": True, "params": params,
                "cv": scores, "pipeline": search.best_estimator_,
            }  # fmt: skip
        else:
            pipeline = build_pipeline(name, strategy, cfg, y_train).fit(X_train, y_train)
            candidates[name] = {
                "strategy": strategy, "tuned": False, "params": "défaut",
                "cv": {k: float(default[k]) for k in scores}, "pipeline": pipeline,
            }  # fmt: skip

    # Étape C : chaque candidat (entraîné sur tout le train) est évalué sur la validation,
    # au seuil qui maximise la précision avec un rappel >= cible.
    val_results = {}
    for name, cand in candidates.items():
        y_proba = cand["pipeline"].predict_proba(X_val)[:, 1]
        chosen = select_threshold(y_val, y_proba, target_recall)
        metrics = compute_metrics(y_val, y_proba, threshold=chosen["threshold"])
        cand.update({"val_metrics": metrics, "val_proba": y_proba})
        val_results[name] = metrics
        with mlflow.start_run(run_name=f"val_{name}_{cand['strategy']}"):
            mlflow.set_tags({"stage": "validation", "model": name, "strategy": cand["strategy"]})
            mlflow.log_params({"model": name, "strategy": cand["strategy"], "tuned": cand["tuned"]})
            mlflow.log_metrics({**cand["cv"], **_flat_val_metrics(metrics)})
        rows.append(
            {
                "stage": "C_validation",
                "model": name,
                "strategy": cand["strategy"],
                "tuned": cand["tuned"],
                **cand["cv"],
                **_flat_val_metrics(metrics),
            }
        )

    # Étape D : modèle final = meilleure précision (à rappel >= cible) sur la validation.
    final_name = max(val_results, key=lambda n: val_results[n]["precision"])
    final = candidates[final_name]
    comparison = pd.DataFrame(rows)
    save_comparison(comparison, reports)
    print(f"\nValidation, au seuil choisi pour un rappel >= {target_recall:.0%} :")
    print(metrics_table(val_results).to_string(float_format=lambda x: f"{x:.3f}"))

    # Analyse de coût (hypothèses illustratives) et figures, sur la validation.
    params = cost_parameters(cfg, splits["train"])
    threshold = final["val_metrics"]["threshold"]
    costs = cost_analysis(y_val, final["val_proba"], params, threshold)
    figures = reports / "figures"
    plot_threshold_analysis(
        y_val,
        final["val_proba"],
        select_threshold(y_val, final["val_proba"], target_recall),
        target_recall,
        figures / "threshold_validation.png",
    )
    plot_cost_curve(
        cost_curve(y_val, final["val_proba"], params), costs, figures / "cost_validation.png"
    )
    (reports / "cost_analysis_validation.json").write_text(
        json.dumps(costs, indent=2), encoding="utf-8"
    )

    artifact_path, metadata = save_final_model(final_name, final, cfg, costs)
    with mlflow.start_run(run_name=f"final_{final_name}") as run:
        mlflow.set_tags({"stage": "final", "model": final_name, "strategy": final["strategy"]})
        mlflow.log_params(
            {"model": final_name, "strategy": final["strategy"], "threshold": threshold}
            | (final["params"] if isinstance(final["params"], dict) else {})
        )
        mlflow.log_metrics({**final["cv"], **_flat_val_metrics(final["val_metrics"])})
        mlflow.log_artifact(str(artifact_path))
        mlflow.log_artifact(str(figures / "threshold_validation.png"))
        mlflow.log_artifact(str(figures / "cost_validation.png"))
        metadata["mlflow_run_id"] = run.info.run_id
    write_metadata(metadata, cfg)
    print(f"\nModèle final : {final_name} ({final['strategy']}), seuil = {threshold:.4f}")
    print(f"Sauvegardé dans {artifact_path}")
    return metadata


def save_comparison(comparison: pd.DataFrame, reports_dir: Path) -> None:
    """Tableau récapitulatif des expériences (CSV + Markdown)."""
    comparison.to_csv(reports_dir / "model_comparison.csv", index=False)
    cols = [
        "stage", "model", "strategy", "cv_pr_auc_mean", "cv_pr_auc_std", "cv_roc_auc_mean",
        "cv_precision_at_recall_mean", "val_threshold", "val_precision", "val_recall",
        "val_f1", "val_pr_auc", "val_roc_auc",
    ]  # fmt: skip
    table = comparison.reindex(columns=cols)
    (reports_dir / "model_comparison.md").write_text(to_markdown(table), encoding="utf-8")


def to_markdown(df: pd.DataFrame) -> str:
    """Tableau Markdown simple (évite une dépendance à tabulate)."""

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return "" if np.isnan(v) else f"{v:.3f}"
        return str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def save_final_model(
    name: str, final: dict[str, Any], cfg: dict[str, Any], costs: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    """Un seul artefact (pipeline + seuil + variables) et ses métadonnées."""
    path = resolve_path(cfg["model"]["artifact"])
    path.parent.mkdir(parents=True, exist_ok=True)
    threshold = final["val_metrics"]["threshold"]
    artifact = {
        "pipeline": final["pipeline"],
        "threshold": threshold,
        "model_name": name,
        "strategy": final["strategy"],
        "version": cfg["model"]["version"],
        "numeric_features": cfg["columns"]["numeric"],
        "categorical_features": cfg["columns"]["categorical"],
    }
    joblib.dump(artifact, path)
    metadata = {
        "version": cfg["model"]["version"],
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": name,
        "strategy": final["strategy"],
        "tuned": final["tuned"],
        "params": final["params"],
        "threshold": threshold,
        "threshold_rule": (
            f"précision maximale parmi les seuils de rappel >= "
            f"{cfg['threshold']['target_recall']:.0%} (validation)"
        ),
        "seed": cfg["seed"],
        "trained_on": "train uniquement",
        "cv_metrics": final["cv"],
        "validation_metrics": final["val_metrics"],
        "validation_cost_analysis": costs,
        "test_metrics": None,
        "libraries": {
            lib: version(lib)
            for lib in ("scikit-learn", "imbalanced-learn", "xgboost", "pandas", "numpy")
        },
        "artifact": path.relative_to(PROJECT_ROOT).as_posix(),
    }
    return path, metadata


def write_metadata(metadata: dict[str, Any], cfg: dict[str, Any]) -> None:
    path = resolve_path(cfg["model"]["metadata"])
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def main_baseline() -> None:
    cfg = load_config()
    results = train_baselines(cfg)

    table = metrics_table(results)
    print("Baselines — métriques sur la VALIDATION (seuil 0,5) :\n")
    print(table.to_string(float_format=lambda x: f"{x:.3f}"))

    out = resolve_path(cfg["paths"]["reports_dir"]) / "baseline_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nMétriques enregistrées dans {out}")


def main() -> None:
    import sys

    if "--baseline" in sys.argv:
        main_baseline()
    else:
        run_experiments(force="--force" in sys.argv)


if __name__ == "__main__":
    main()

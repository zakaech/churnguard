"""Tests des pipelines de modélisation, de la sauvegarde du modèle et du garde-fou du test."""

import copy

import joblib
import numpy as np
import pytest

from churnguard import train as train_module
from churnguard.data import clean, split_data, split_xy
from churnguard.evaluate import evaluate_on_test
from churnguard.predict import load_model, predict, predict_proba
from churnguard.train import (
    MODELS,
    STRATEGIES,
    build_pipeline,
    check_final_model_not_tested,
    cross_validate_pipeline,
    run_experiments,
    search_cv_metrics,
    search_hyperparameters,
)


@pytest.fixture
def data(raw_sample, cfg):
    splits = split_data(clean(raw_sample, cfg), cfg)
    X_train, y_train = split_xy(splits["train"], cfg)
    X_val, y_val = split_xy(splits["val"], cfg)
    return X_train, y_train, X_val, y_val


@pytest.fixture
def small_cfg(cfg):
    """Configuration allégée pour des tests rapides sur l'échantillon."""
    small = copy.deepcopy(cfg)
    small["cv"]["n_splits"] = 3
    small["search"]["n_iter"] = 2
    return small


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("name", MODELS)
def test_pipeline_predicts_one_probability_per_row(name, strategy, data, cfg):
    X_train, y_train, X_val, _ = data
    pipeline = build_pipeline(name, strategy, cfg, y_train).fit(X_train, y_train)
    proba = pipeline.predict_proba(X_val)[:, 1]
    # SMOTENC n'agit qu'au fit : à la prédiction, autant de lignes en sortie qu'en entrée.
    assert proba.shape == (len(X_val),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_smotenc_resamples_training_data_only(data, cfg):
    X_train, y_train, _, _ = data
    pipeline = build_pipeline("logistic_regression", "smotenc", cfg, y_train)
    X_res, y_res = pipeline.named_steps["smotenc"].fit_resample(X_train, y_train)
    assert (y_res == 1).sum() == (y_res == 0).sum() == (y_train == 0).sum()
    assert set(X_res["Contract"]) <= set(X_train["Contract"])


def test_unknown_strategy_raises(data, cfg):
    _, y_train, _, _ = data
    with pytest.raises(ValueError):
        build_pipeline("xgboost", "oversample_everything", cfg, y_train)


def test_cross_validation_returns_mean_and_std(data, small_cfg):
    X_train, y_train, _, _ = data
    pipeline = build_pipeline("logistic_regression", "class_weight", small_cfg, y_train)
    scores = cross_validate_pipeline(pipeline, X_train, y_train, small_cfg)
    for metric in ("pr_auc", "roc_auc", "precision_at_recall"):
        assert 0 <= scores[f"cv_{metric}_mean"] <= 1
        assert scores[f"cv_{metric}_std"] >= 0


def test_search_runs_and_reports_best_metrics(data, small_cfg):
    X_train, y_train, _, _ = data
    search = search_hyperparameters("logistic_regression", "none", X_train, y_train, small_cfg)
    scores = search_cv_metrics(search)
    assert scores["cv_pr_auc_mean"] == pytest.approx(search.best_score_)


def test_saved_model_roundtrip(data, cfg, tmp_path):
    X_train, y_train, X_val, _ = data
    pipeline = build_pipeline("xgboost", "class_weight", cfg, y_train).fit(X_train, y_train)
    artifact = {
        "pipeline": pipeline,
        "threshold": 0.3,
        "model_name": "xgboost",
        "numeric_features": cfg["columns"]["numeric"],
        "categorical_features": cfg["columns"]["categorical"],
    }
    path = tmp_path / "model.joblib"
    joblib.dump(artifact, path)
    loaded = load_model(path)

    np.testing.assert_array_equal(predict_proba(loaded, X_val), pipeline.predict_proba(X_val)[:, 1])
    assert loaded["threshold"] == 0.3
    out = predict(loaded, X_val)
    assert (out["churn_predicted"] == (out["churn_probability"] >= 0.3)).all()


def test_training_refuses_to_overwrite_tested_model(cfg, tmp_path, monkeypatch):
    guarded = copy.deepcopy(cfg)
    guarded["paths"]["reports_dir"] = str(tmp_path)
    (tmp_path / "test_metrics.json").write_text("{}", encoding="utf-8")

    # Le refus doit intervenir avant tout chargement de données ou entraînement.
    def fail(*args, **kwargs):
        raise AssertionError("les données ne doivent pas être chargées")

    monkeypatch.setattr(train_module, "get_splits", fail)
    with pytest.raises(RuntimeError, match="obsolètes"):
        run_experiments(guarded)


def test_training_guard_allows_force_and_untested_model(cfg, tmp_path, capsys):
    guarded = copy.deepcopy(cfg)
    guarded["paths"]["reports_dir"] = str(tmp_path)
    check_final_model_not_tested(guarded)  # pas de rapport de test : rien à protéger

    (tmp_path / "test_metrics.json").write_text("{}", encoding="utf-8")
    check_final_model_not_tested(guarded, force=True)
    assert "obsolètes" in capsys.readouterr().out


def test_test_evaluation_refuses_second_run(cfg, tmp_path):
    guarded = copy.deepcopy(cfg)
    guarded["paths"]["reports_dir"] = str(tmp_path)
    (tmp_path / "test_metrics.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="déjà été évalué"):
        evaluate_on_test(guarded)

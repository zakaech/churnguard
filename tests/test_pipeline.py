"""Tests du prétraitement, des métriques et des baselines (sur l'échantillon synthétique)."""

import numpy as np
import pytest

from churnguard.data import clean, split_data, split_xy
from churnguard.evaluate import compute_metrics, metrics_table
from churnguard.features import build_preprocessor
from churnguard.train import build_baselines


@pytest.fixture
def splits(raw_sample, cfg):
    return split_data(clean(raw_sample, cfg), cfg)


def test_preprocessor_learns_on_train_only(splits, cfg):
    X_train, _ = split_xy(splits["train"], cfg)
    X_val, _ = split_xy(splits["val"], cfg)
    pre = build_preprocessor(cfg).fit(X_train)

    scaler = pre.named_transformers_["num"]
    np.testing.assert_allclose(scaler.mean_, X_train[cfg["columns"]["numeric"]].mean())

    out_train, out_val = pre.transform(X_train), pre.transform(X_val)
    assert out_train.shape[1] == out_val.shape[1]
    assert out_val.shape[0] == len(X_val)
    assert not np.isnan(out_val).any()


def test_preprocessor_ignores_unknown_category(splits, cfg):
    X_train, _ = split_xy(splits["train"], cfg)
    X_val, _ = split_xy(splits["val"], cfg)
    pre = build_preprocessor(cfg).fit(X_train)

    X_new = X_val.head(1).copy()
    X_new["PaymentMethod"] = "Crypto"
    assert pre.transform(X_new).shape == (1, pre.transform(X_val).shape[1])


def test_compute_metrics_known_values():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.6, 0.4, 0.9])
    m = compute_metrics(y_true, y_proba, threshold=0.5)

    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["roc_auc"] == pytest.approx(0.75)


def test_threshold_changes_decisions_not_auc():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.6, 0.4, 0.9])
    low, high = compute_metrics(y_true, y_proba, 0.3), compute_metrics(y_true, y_proba, 0.7)

    assert low["recall"] == pytest.approx(1.0)
    assert high["recall"] == pytest.approx(0.5)
    assert low["roc_auc"] == high["roc_auc"]
    assert low["pr_auc"] == high["pr_auc"]


def test_baselines_fit_and_predict_probabilities(splits, cfg):
    X_train, y_train = split_xy(splits["train"], cfg)
    X_val, y_val = split_xy(splits["val"], cfg)

    results = {}
    for name, pipeline in build_baselines(cfg).items():
        proba = pipeline.fit(X_train, y_train).predict_proba(X_val)[:, 1]
        assert proba.shape == (len(X_val),)
        assert ((proba >= 0) & (proba <= 1)).all()
        results[name] = compute_metrics(y_val, proba)

    assert results["dummy_most_frequent"]["recall"] == 0.0
    table = metrics_table(results)
    assert list(table.index) == ["dummy_most_frequent", "logistic_regression"]
    assert table["tp"].dtype.kind == "i"

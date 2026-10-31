"""Tests des explications SHAP (petit Random Forest entraîné sur l'échantillon synthétique)."""

import numpy as np
import pandas as pd
import pytest

from churnguard.data import clean, split_data, split_xy
from churnguard.explain import ChurnExplainer, value_label
from churnguard.predict import predict_proba
from churnguard.train import build_pipeline


@pytest.fixture(scope="module")
def setup(request):
    from churnguard.config import load_config

    cfg = load_config()
    raw = pd.read_csv(request.path.parent / "data" / "sample.csv")
    splits = split_data(clean(raw, cfg), cfg)
    X_train, y_train = split_xy(splits["train"], cfg)
    pipeline = build_pipeline("random_forest", "none", cfg, y_train).fit(X_train, y_train)
    artifact = {
        "pipeline": pipeline,
        "threshold": 0.3,
        "numeric_features": cfg["columns"]["numeric"],
        "categorical_features": cfg["columns"]["categorical"],
    }
    return cfg, artifact, ChurnExplainer(artifact), splits["val"]


def test_contributions_are_additive(setup):
    _, artifact, explainer, val = setup
    contrib = explainer.contributions(val)
    np.testing.assert_allclose(
        explainer.base_value + contrib.sum(axis=1).to_numpy(),
        predict_proba(artifact, val),
        atol=1e-9,
    )


def test_contributions_grouped_by_original_feature(setup):
    cfg, _, explainer, val = setup
    contrib = explainer.contributions(val)
    assert list(contrib.columns) == cfg["columns"]["numeric"] + cfg["columns"]["categorical"]
    assert len(contrib.columns) == 19

    raw = explainer.transformed_shap_values(val)
    np.testing.assert_allclose(contrib.sum(axis=1).to_numpy(), raw.sum(axis=1), atol=1e-12)
    contract = explainer.groups["Contract"]
    assert len(contract) == 3  # 3 modalités, pas de colonne supprimée (non binaire)
    np.testing.assert_allclose(contrib["Contract"], raw[:, contract].sum(axis=1), atol=1e-12)


def test_top_factors_sorted_signed_and_readable(setup):
    _, _, explainer, val = setup
    client = val.head(1)
    factors = explainer.top_factors(client, k=5)

    assert len(factors) == 5
    impacts = [abs(f["impact"]) for f in factors]
    assert impacts == sorted(impacts, reverse=True)
    for f in factors:
        assert f["direction"] == ("hausse" if f["impact"] > 0 else "baisse")
        assert ("augmente" in f["text"]) == (f["impact"] > 0)
        assert f["text"].startswith(f["label"])
        # Valeur réelle du client (non normalisée).
        assert f["value"] == client.iloc[0][f["feature"]]


def test_top_factors_accepts_series_and_rejects_several_clients(setup):
    _, _, explainer, val = setup
    from_series = explainer.top_factors(val.iloc[0], k=3)
    from_frame = explainer.top_factors(val.head(1), k=3)
    assert [f["feature"] for f in from_series] == [f["feature"] for f in from_frame]
    with pytest.raises(ValueError):
        explainer.top_factors(val.head(2))


def test_explanation_is_deterministic(setup):
    _, _, explainer, val = setup
    assert explainer.top_factors(val.head(1)) == explainer.top_factors(val.head(1))


def test_unknown_category_and_missing_value(setup):
    _, _, explainer, val = setup
    client = val.head(1).copy()
    client["PaymentMethod"] = "Crypto"
    client["MonthlyCharges"] = np.nan
    with pytest.warns(UserWarning, match="unknown categories"):
        factors = explainer.top_factors(client, k=19)
    by_feature = {f["feature"]: f for f in factors}
    assert by_feature["MonthlyCharges"]["value_label"] == "non renseigné"
    assert by_feature["PaymentMethod"]["value_label"] == "Crypto"


def test_global_importance_sorted(setup):
    _, _, explainer, val = setup
    importance = explainer.global_importance(val)
    assert len(importance) == 19
    assert importance["mean_abs_impact"].is_monotonic_decreasing
    np.testing.assert_allclose(importance["mean_abs_points"], 100 * importance["mean_abs_impact"])


def test_value_labels():
    assert value_label("Contract", "Month-to-month") == "au mois"
    assert value_label("tenure", 12) == "12 mois"
    assert value_label("MonthlyCharges", 1234.5) == "1 234,50"
    assert value_label("TechSupport", "No internet service") == "pas d'internet"
    assert value_label("Partner", "Yes") == "oui"


def test_non_random_forest_model_rejected(setup):
    cfg, artifact, _, val = setup
    X, y = split_xy(val, cfg)
    other = dict(artifact, pipeline=build_pipeline("logistic_regression", "none", cfg, y).fit(X, y))
    with pytest.raises(NotImplementedError):
        ChurnExplainer(other)


def test_top_factors_batch_matches_single_client(setup):
    _, _, explainer, val = setup
    batch = explainer.top_factors_batch(val.head(4), k=5)
    assert len(batch) == 4
    for i, factors in enumerate(batch):
        assert factors == explainer.top_factors(val.iloc[[i]], k=5)

"""Tests du nettoyage et du découpage (sur l'échantillon synthétique)."""

import pandas as pd
import pytest

from churnguard.data import (
    clean,
    feature_columns,
    load_splits,
    save_splits,
    split_data,
    split_xy,
)


def test_total_charges_numeric_without_missing(raw_sample, cfg):
    df = clean(raw_sample, cfg)
    assert pd.api.types.is_float_dtype(df["TotalCharges"])
    assert df["TotalCharges"].notna().all()


def test_blank_total_charges_set_to_zero_for_new_clients(raw_sample, cfg):
    blank = raw_sample["TotalCharges"].str.strip() == ""
    assert blank.any(), "l'échantillon doit contenir des TotalCharges vides"
    df = clean(raw_sample, cfg)
    assert (df.loc[blank, "TotalCharges"] == 0).all()
    assert (df.loc[blank, "tenure"] == 0).all()


def test_blank_total_charges_with_positive_tenure_raises(raw_sample, cfg):
    bad = raw_sample.copy()
    idx = bad.index[bad["tenure"] > 0][0]
    bad.loc[idx, "TotalCharges"] = " "
    with pytest.raises(ValueError):
        clean(bad, cfg)


def test_non_numeric_total_charges_raises(raw_sample, cfg):
    bad = raw_sample.copy()
    bad.loc[bad.index[0], "TotalCharges"] = "abc"
    with pytest.raises(ValueError):
        clean(bad, cfg)


def test_target_is_binary(raw_sample, cfg):
    df = clean(raw_sample, cfg)
    assert set(df["Churn"].unique()) <= {0, 1}
    assert (df["Churn"] == 1).sum() == (raw_sample["Churn"] == "Yes").sum()


def test_unexpected_target_value_raises(raw_sample, cfg):
    bad = raw_sample.copy()
    bad.loc[bad.index[0], "Churn"] = "Maybe"
    with pytest.raises(ValueError):
        clean(bad, cfg)


def test_senior_citizen_mapped_to_yes_no(raw_sample, cfg):
    df = clean(raw_sample, cfg)
    assert set(df["SeniorCitizen"].unique()) <= {"Yes", "No"}


def test_clean_does_not_modify_input(raw_sample, cfg):
    before = raw_sample.copy()
    clean(raw_sample, cfg)
    pd.testing.assert_frame_equal(raw_sample, before)


def test_features_exclude_id_and_target(raw_sample, cfg):
    X, y = split_xy(clean(raw_sample, cfg), cfg)
    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert list(X.columns) == feature_columns(cfg)
    assert len(X) == len(y)


def test_split_sizes_and_disjoint(raw_sample, cfg):
    df = clean(raw_sample, cfg)
    splits = split_data(df, cfg)
    ids = {name: set(part["customerID"]) for name, part in splits.items()}

    assert sum(len(p) for p in splits.values()) == len(df)
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["val"].isdisjoint(ids["test"])
    assert len(splits["train"]) == pytest.approx(0.6 * len(df), abs=2)
    assert len(splits["val"]) == pytest.approx(0.2 * len(df), abs=2)
    assert len(splits["test"]) == pytest.approx(0.2 * len(df), abs=2)


def test_split_is_stratified(raw_sample, cfg):
    df = clean(raw_sample, cfg)
    overall = df["Churn"].mean()
    for part in split_data(df, cfg).values():
        # Tolérance : une ligne de différence sur une partie de ~60 lignes.
        assert part["Churn"].mean() == pytest.approx(overall, abs=0.02)


def test_split_is_reproducible(raw_sample, cfg):
    df = clean(raw_sample, cfg)
    first, second = split_data(df, cfg), split_data(df, cfg)
    for name in first:
        assert first[name]["customerID"].tolist() == second[name]["customerID"].tolist()


def test_save_and_load_splits_roundtrip(raw_sample, cfg, tmp_path):
    splits = split_data(clean(raw_sample, cfg), cfg)
    save_splits(splits, tmp_path)
    loaded = load_splits(tmp_path, cfg)
    for name, part in splits.items():
        pd.testing.assert_frame_equal(
            loaded[name].reset_index(drop=True),
            part.reset_index(drop=True),
            check_dtype=False,
        )
        assert pd.api.types.is_float_dtype(loaded[name]["TotalCharges"])

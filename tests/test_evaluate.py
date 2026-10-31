"""Tests du choix du seuil et de l'analyse de coût (cas simples à résultat connu)."""

import numpy as np
import pandas as pd
import pytest

from churnguard.evaluate import (
    business_cost,
    compute_metrics,
    cost_analysis,
    cost_parameters,
    precision_at_recall,
    select_threshold,
)

# 4 départs (probas 0.4, 0.6, 0.7, 0.8) et 4 clients restés (0.1, 0.2, 0.3, 0.65).
Y = np.array([0, 0, 0, 1, 1, 1, 1, 0])
P = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.65])


def test_select_threshold_known_case():
    # Rappel >= 75 % : seuil 0.6 -> P = 3/4, R = 3/4 ; seuil 0.4 -> P = 4/5, R = 1 ;
    # seuil 0.3 -> P = 4/6. La meilleure précision est au seuil 0.4.
    chosen = select_threshold(Y, P, target_recall=0.75)
    assert chosen["threshold"] == pytest.approx(0.4)
    assert chosen["precision"] == pytest.approx(0.8)
    assert chosen["recall"] == pytest.approx(1.0)


def test_select_threshold_consistent_with_metrics():
    chosen = select_threshold(Y, P, target_recall=0.75)
    m = compute_metrics(Y, P, threshold=chosen["threshold"])
    assert m["recall"] >= 0.75
    assert m["precision"] == pytest.approx(chosen["precision"])


def test_select_threshold_ties_keep_highest_threshold():
    # Seuils 0.9 (P = 1, R = 0.5) et 0.5 (P = 1, R = 1) : égalité de précision -> 0.9.
    y, p = np.array([0, 1, 1]), np.array([0.2, 0.5, 0.9])
    assert select_threshold(y, p, target_recall=0.5)["threshold"] == pytest.approx(0.9)


def test_select_threshold_unreachable_recall_raises():
    with pytest.raises(ValueError):
        select_threshold(Y, P, target_recall=1.01)


def test_precision_at_recall():
    assert precision_at_recall(Y, P, 0.75) == pytest.approx(0.8)


def test_business_cost_known_case():
    # 2 × 100 + 3 × 0.7 × 100 + (3 + 5) × 10 = 200 + 210 + 80 = 490
    cm = {"tn": 10, "fp": 5, "fn": 2, "tp": 3}
    cost = business_cost(cm, lost_customer_cost=100, offer_cost=10, offer_success_rate=0.3)
    assert cost == pytest.approx(490)


def test_business_cost_without_offers_equals_all_churners_lost():
    cm = {"tn": 10, "fp": 0, "fn": 4, "tp": 0}
    assert business_cost(cm, 100, 10, 0.3) == pytest.approx(400)


def test_cost_parameters_from_train_median(cfg):
    train = pd.DataFrame({"MonthlyCharges": [10.0, 50.0, 70.0]})
    params = cost_parameters(cfg, train)
    assert params["median_monthly_charges"] == pytest.approx(50.0)
    assert params["lost_customer_cost"] == pytest.approx(
        cfg["cost_analysis"]["lost_customer_months"] * 50
    )
    assert params["offer_cost"] == pytest.approx(cfg["cost_analysis"]["offer_months"] * 50)


def test_cost_analysis_min_cost_not_above_other_thresholds():
    params = {"lost_customer_cost": 100, "offer_cost": 10, "offer_success_rate": 0.3}
    analysis = cost_analysis(Y, P, params, chosen_threshold=0.4)
    assert analysis["no_action_cost"] == pytest.approx(400)
    assert analysis["min_cost_threshold"]["cost"] <= analysis["chosen_threshold"]["cost"]
    assert analysis["min_cost_threshold"]["cost"] <= analysis["threshold_0_5"]["cost"]

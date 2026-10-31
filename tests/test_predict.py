"""Tests des niveaux de risque."""

import numpy as np
import pytest

from churnguard.predict import RISK_LEVELS, risk_level

THRESHOLD, MEDIUM_FROM = 0.2712, 0.05


@pytest.mark.parametrize(
    ("proba", "expected"),
    [
        (THRESHOLD, "élevé"),
        (np.nextafter(THRESHOLD, 0), "moyen"),
        (MEDIUM_FROM, "moyen"),
        (np.nextafter(MEDIUM_FROM, 0), "faible"),
        (0.0, "faible"),
        (1.0, "élevé"),
    ],
)
def test_risk_level_boundaries(proba, expected):
    assert risk_level(np.array([proba]), THRESHOLD, MEDIUM_FROM)[0] == expected


def test_high_risk_exactly_matches_flagged_clients():
    rng = np.random.default_rng(0)
    proba = np.concatenate([rng.uniform(size=10_000), [THRESHOLD, MEDIUM_FROM]])
    levels = risk_level(proba, THRESHOLD, MEDIUM_FROM)
    np.testing.assert_array_equal(levels == "élevé", proba >= THRESHOLD)
    assert set(levels) <= set(RISK_LEVELS)


@pytest.mark.parametrize("medium_from", [0.0, THRESHOLD, 0.5, -0.1])
def test_inconsistent_medium_bound_rejected(medium_from):
    with pytest.raises(ValueError):
        risk_level(np.array([0.1]), THRESHOLD, medium_from)

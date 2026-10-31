"""Fixtures partagées : configuration et échantillon synthétique (jamais les vraies données)."""

from pathlib import Path

import pandas as pd
import pytest

from churnguard.config import load_config

SAMPLE_PATH = Path(__file__).parent / "data" / "sample.csv"


@pytest.fixture
def cfg() -> dict:
    return load_config()


@pytest.fixture
def raw_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)

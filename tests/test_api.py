"""Tests de l'API FastAPI, avec un petit modèle entraîné sur l'échantillon synthétique.

Ils ne dépendent ni du vrai modèle ni des données Kaggle.
"""

import io

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from churnguard.api.main import create_app
from churnguard.api.schemas import EXAMPLE_CUSTOMER, EXAMPLE_CUSTOMER_2
from churnguard.config import load_config
from churnguard.data import clean, split_data, split_xy
from churnguard.train import build_pipeline

SAMPLE = "tests/data/sample.csv"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    cfg = load_config()
    raw = pd.read_csv(SAMPLE)
    X, y = split_xy(split_data(clean(raw, cfg), cfg)["train"], cfg)
    artifact = {
        "pipeline": build_pipeline("random_forest", "none", cfg, y).fit(X, y),
        "threshold": 0.3,
        "model_name": "random_forest",
        "version": "test",
        "numeric_features": cfg["columns"]["numeric"],
        "categorical_features": cfg["columns"]["categorical"],
    }
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump(artifact, path)
    with TestClient(create_app(path)) as c:
        yield c


def customer(**changes):
    return EXAMPLE_CUSTOMER | changes


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# --- /health et /predict -------------------------------------------------------------------


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "test"
    assert body["threshold"] == 0.3
    assert 0 < body["risk_medium_from"] < body["threshold"]


def test_predict_valid_customer(client):
    r = client.post("/predict", json=customer())
    assert r.status_code == 200
    body = r.json()
    assert body["customerID"] == "FICTIF-A"
    assert 0 <= body["churn_probability"] <= 1
    assert body["churn_predicted"] == (body["churn_probability"] >= body["threshold"])
    assert (body["risk_level"] == "élevé") == body["churn_predicted"]
    assert len(body["top_factors"]) == 5
    assert all(f["direction"] in ("hausse", "baisse") for f in body["top_factors"])


def test_predict_is_stable(client):
    first = client.post("/predict", json=customer()).json()
    second = client.post("/predict", json=customer()).json()
    assert first == second


def test_predict_accepts_blank_total_charges_for_new_customer(client):
    r = client.post("/predict", json=customer(tenure=0, TotalCharges=None))
    assert r.status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {k: v for k, v in EXAMPLE_CUSTOMER.items() if k != "Contract"},  # champ manquant
        customer(Contract="Weekly"),  # modalité inconnue
        customer(tenure=-1),  # ancienneté négative
        customer(tenure=2.5),  # ancienneté non entière
        customer(MonthlyCharges=0),  # montant nul
        customer(SeniorCitizen=2),
        customer(SeniorCitizen=True),  # booléen refusé
        customer(TotalCharges=None),  # vide alors que tenure > 0
        customer(InternetService="No"),  # options internet incohérentes
        customer(OnlineSecurity="No internet service"),  # option incohérente avec la fibre
        customer(PhoneService="No"),  # MultipleLines incohérent
        customer(Churn="Yes"),  # champ en trop
    ],
)
def test_predict_rejects_invalid_input(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_consistency_error_message_in_french(client):
    r = client.post("/predict", json=customer(TotalCharges=None))
    assert "ancienneté 0" in str(r.json()["detail"])


# --- /predict/batch ------------------------------------------------------------------------


def test_batch_keeps_order_and_counts(client):
    customers = [customer(), EXAMPLE_CUSTOMER_2, customer(customerID="X-3", tenure=40)]
    r = client.post("/predict/batch", json={"customers": customers})
    assert r.status_code == 200
    body = r.json()
    assert body["n_customers"] == 3
    assert [p["customerID"] for p in body["predictions"]] == ["FICTIF-A", "FICTIF-C", "X-3"]
    assert sum(body["risk_counts"].values()) == 3
    assert body["include_factors"] is False
    assert all(p["top_factors"] is None for p in body["predictions"])


def test_batch_matches_single_prediction(client):
    single = client.post("/predict", json=customer()).json()
    batch = client.post("/predict/batch", json={"customers": [customer()]}).json()
    assert batch["predictions"][0]["churn_probability"] == single["churn_probability"]


def test_batch_include_factors(client):
    r = client.post(
        "/predict/batch",
        params={"include_factors": True},
        json={"customers": [customer(), EXAMPLE_CUSTOMER_2]},
    )
    assert r.status_code == 200
    assert all(len(p["top_factors"]) == 5 for p in r.json()["predictions"])


def test_batch_rejects_empty_and_invalid(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422
    bad = {"customers": [customer(), customer(Contract="Weekly")]}
    assert client.post("/predict/batch", json=bad).status_code == 422


def test_batch_too_large(client):
    service = client.app.state.service
    original = service.max_batch_size, service.max_batch_size_with_factors
    service.max_batch_size, service.max_batch_size_with_factors = 3, 2
    try:
        body = {"customers": [customer()] * 3}
        assert client.post("/predict/batch", json=body).status_code == 200
        r = client.post("/predict/batch", params={"include_factors": True}, json=body)
        assert r.status_code == 413
        assert "include_factors" in r.json()["detail"]
        body = {"customers": [customer()] * 4}
        assert client.post("/predict/batch", json=body).status_code == 413
    finally:
        service.max_batch_size, service.max_batch_size_with_factors = original


# --- /predict/batch/csv --------------------------------------------------------------------


def test_csv_raw_format(client):
    raw = pd.read_csv(SAMPLE)
    r = client.post(
        "/predict/batch/csv", files={"file": ("sample.csv", csv_bytes(raw), "text/csv")}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_customers"] == len(raw)
    assert body["ignored_columns"] == ["Churn"]
    assert [p["customerID"] for p in body["predictions"]] == raw["customerID"].tolist()


def test_csv_with_factors(client):
    raw = pd.read_csv(SAMPLE).head(3)
    r = client.post(
        "/predict/batch/csv",
        params={"include_factors": True},
        files={"file": ("s.csv", csv_bytes(raw), "text/csv")},
    )
    assert all(len(p["top_factors"]) == 5 for p in r.json()["predictions"])


def test_csv_missing_column(client):
    raw = pd.read_csv(SAMPLE).drop(columns=["Contract"])
    r = client.post("/predict/batch/csv", files={"file": ("s.csv", csv_bytes(raw), "text/csv")})
    assert r.status_code == 422
    assert "Contract" in r.json()["detail"]


def test_csv_invalid_row_reports_line_number(client):
    raw = pd.read_csv(SAMPLE).head(5)
    raw.loc[3, "Contract"] = "Weekly"
    r = client.post("/predict/batch/csv", files={"file": ("s.csv", csv_bytes(raw), "text/csv")})
    assert r.status_code == 422
    lines = r.json()["detail"]["lignes"]
    assert [entry["ligne"] for entry in lines] == [5]  # en-tête = ligne 1, index 3 -> ligne 5


def test_csv_unreadable(client):
    r = client.post(
        "/predict/batch/csv", files={"file": ("s.csv", io.BytesIO(b"\xff\xfe\x00"), "text/csv")}
    )
    assert r.status_code == 422

"""Tests de l'interface : fonctions d'aide, client HTTP (transport simulé) et application
Streamlit (AppTest, API simulée). Aucun réseau ni modèle nécessaire."""

import json

import httpx
import pandas as pd
import pytest

from churnguard.api.schemas import EXAMPLE_CUSTOMER
from churnguard.ui.client import (
    INTERNET_OPTIONS,
    ApiClient,
    ApiError,
    build_customer,
    chunk_ranges,
    format_detail,
    results_table,
    risk_distribution,
)

PRED = {
    "customerID": "A",
    "churn_probability": 0.4,
    "churn_predicted": True,
    "risk_level": "élevé",
    "top_factors": None,
}


def test_build_customer_enforces_consistency():
    values = EXAMPLE_CUSTOMER | {
        "InternetService": "No",
        "PhoneService": "No",
        "SeniorCitizen": True,
    }
    customer = build_customer(values)
    assert all(customer[o] == "No internet service" for o in INTERNET_OPTIONS)
    assert customer["MultipleLines"] == "No phone service"
    assert customer["SeniorCitizen"] == 1


def test_build_customer_blank_total_for_new_client():
    customer = build_customer(EXAMPLE_CUSTOMER | {"tenure": 0, "TotalCharges": ""})
    assert customer["TotalCharges"] is None


def test_chunk_ranges():
    assert list(chunk_ranges(5, 2)) == [(0, 2), (2, 4), (4, 5)]
    assert list(chunk_ranges(0, 3)) == []
    with pytest.raises(ValueError):
        list(chunk_ranges(5, 0))


def test_results_table_sorted_by_risk():
    preds = [
        PRED | {"customerID": "low", "churn_probability": 0.01, "risk_level": "faible"},
        PRED,
        PRED | {"customerID": "top", "churn_probability": 0.9},
    ]
    table = results_table(preds)
    assert table["customerID"].tolist() == ["top", "A", "low"]
    assert "facteurs" not in table.columns


def test_risk_distribution_has_all_levels():
    dist = risk_distribution([PRED, PRED | {"risk_level": "faible"}])
    assert dist["niveau_risque"].tolist() == ["élevé", "moyen", "faible"]
    assert dist["clients"].tolist() == [1, 0, 1]
    assert dist["part"].sum() == pytest.approx(1.0)


def test_format_detail_csv_errors():
    detail = {"message": "1 ligne(s) invalide(s).", "lignes": [{"ligne": 7, "erreurs": ["x"]}]}
    assert "ligne 7 : x" in format_detail(detail)


def make_client(handler) -> ApiClient:
    return ApiClient("http://api.test", api_key="k", transport=httpx.MockTransport(handler))


def test_client_sends_api_key_and_raises_api_error():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-API-Key"] == "k"
        return httpx.Response(401, json={"detail": "Clé d'API manquante ou invalide."})

    with pytest.raises(ApiError) as exc:
        make_client(handler).predict(EXAMPLE_CUSTOMER)
    assert exc.value.status_code == 401


def test_predict_csv_splits_into_batches_and_shifts_line_numbers():
    sizes = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode()
        # Lignes de données du lot (identifiants C0, C1… ; on exclut les en-têtes multipart).
        rows = [line for line in body.splitlines() if line.startswith("C") and "," in line]
        sizes.append(len(rows))
        if any("bad" in r for r in rows):
            return httpx.Response(
                422,
                json={"detail": {"message": "1 ligne", "lignes": [{"ligne": 2, "erreurs": ["e"]}]}},
            )
        return httpx.Response(
            200,
            json={"predictions": [PRED] * len(rows), "ignored_columns": ["Churn"]},
        )

    df = pd.DataFrame({"customerID": [f"C{i}" for i in range(5)], "Churn": ["No"] * 5})
    predictions, ignored = make_client(handler).predict_csv(
        df.to_csv(index=False).encode(), include_factors=False, batch_size=2
    )
    assert sizes == [2, 2, 1]
    assert len(predictions) == 5
    assert ignored == ["Churn"]

    df.loc[3, "customerID"] = "Cbad"  # 2e lot (lignes 4-5 du fichier), 2e ligne de données
    with pytest.raises(ApiError) as exc:
        make_client(handler).predict_csv(df.to_csv(index=False).encode(), False, batch_size=2)
    assert exc.value.detail["lignes"][0]["ligne"] == 4  # ligne 2 du lot + décalage de 2


# --- Application Streamlit (AppTest) --------------------------------------------------------

HEALTH = {
    "status": "ok",
    "model_name": "random_forest",
    "model_version": "test",
    "threshold": 0.27,
    "risk_medium_from": 0.05,
    "max_batch_size": 2000,
    "max_batch_size_with_factors": 500,
}
RESULT = PRED | {
    "threshold": 0.27,
    "top_factors": [
        {
            "feature": "Contract",
            "label": "Contrat",
            "value": "Month-to-month",
            "value_label": "au mois",
            "impact_points": 11.6,
            "direction": "hausse",
            "text": "Contrat : au mois — augmente le risque (+11,6 points)",
        }
    ],
}


@pytest.fixture
def app(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(ApiClient, "health", lambda self: HEALTH)
    monkeypatch.setattr(ApiClient, "predict", lambda self, customer: RESULT)
    return AppTest.from_file("../src/churnguard/ui/app.py", default_timeout=30)


def test_app_client_page_shows_result(app):
    app.run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert any("Risque élevé" in e.value for e in app.error)
    assert any("Contrat : au mois" in m.value for m in app.markdown)


def test_app_disables_internet_options_without_internet(app):
    app.run()
    app.selectbox(key="InternetService").select("No").run()
    assert all(app.selectbox(key=o).disabled for o in INTERNET_OPTIONS)


def test_app_reports_unreachable_api(monkeypatch):
    from streamlit.testing.v1 import AppTest

    def down(self, method, url, **kwargs):
        raise httpx.ConnectError("refusé")

    # Panne simulée au niveau de httpx : on teste le vrai chemin client -> application.
    monkeypatch.setattr(httpx.Client, "request", down)
    at = AppTest.from_file("../src/churnguard/ui/app.py", default_timeout=30).run()
    assert any("API injoignable" in e.value for e in at.error)


def test_json_example_is_valid():
    json.dumps(EXAMPLE_CUSTOMER)


def test_network_error_becomes_api_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refusé", request=request)

    with pytest.raises(ApiError) as exc:
        make_client(handler).health()
    assert exc.value.status_code == 503
    assert "injoignable" in str(exc.value)

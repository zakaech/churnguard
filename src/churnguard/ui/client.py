"""Client HTTP de l'API et fonctions d'aide de l'interface (testables sans Streamlit).

L'interface ne charge jamais le modèle : tout passe par l'API.
"""

from __future__ import annotations

import io
import os
from collections.abc import Iterator
from typing import Any

import httpx
import pandas as pd

INTERNET_OPTIONS = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)
RISK_ORDER = {"élevé": 0, "moyen": 1, "faible": 2}


class ApiError(Exception):
    """Erreur renvoyée par l'API, avec un message lisible."""

    def __init__(self, status_code: int, detail: Any) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Erreur {status_code} : {format_detail(detail)}")


def format_detail(detail: Any) -> str:
    """Message d'erreur de l'API (texte, liste Pydantic ou erreurs CSV par ligne) en texte."""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict) and "lignes" in detail:
        lines = [f"ligne {e['ligne']} : {'; '.join(e['erreurs'])}" for e in detail["lignes"]]
        return detail.get("message", "") + "\n" + "\n".join(lines)
    if isinstance(detail, list):
        return "\n".join(f"{'.'.join(map(str, e.get('loc', [])))} : {e.get('msg')}" for e in detail)
    return str(detail)


def build_customer(values: dict[str, Any]) -> dict[str, Any]:
    """Client au format attendu par l'API, à partir des valeurs du formulaire.

    Applique les règles de cohérence pour éviter un refus 422 : sans internet, toutes les
    options valent « No internet service » ; sans ligne téléphonique, MultipleLines vaut
    « No phone service ». TotalCharges est laissé vide pour un nouveau client sans montant.
    """
    customer = dict(values)
    customer["SeniorCitizen"] = int(bool(customer["SeniorCitizen"]))
    if customer["InternetService"] == "No":
        customer.update(dict.fromkeys(INTERNET_OPTIONS, "No internet service"))
    if customer["PhoneService"] == "No":
        customer["MultipleLines"] = "No phone service"
    if customer.get("TotalCharges") in ("", None) and customer["tenure"] == 0:
        customer["TotalCharges"] = None
    return customer


def chunk_ranges(n_rows: int, size: int) -> Iterator[tuple[int, int]]:
    """Découpe [0, n_rows) en intervalles [début, fin) d'au plus `size` lignes."""
    if size <= 0:
        raise ValueError("La taille des lots doit être positive.")
    for start in range(0, n_rows, size):
        yield start, min(start + size, n_rows)


def results_table(predictions: list[dict[str, Any]]) -> pd.DataFrame:
    """Tableau des résultats, trié du risque le plus élevé au plus faible."""
    rows = []
    for p in predictions:
        row = {
            "customerID": p.get("customerID"),
            "probabilite_depart": round(p["churn_probability"], 4),
            "niveau_risque": p["risk_level"],
            "signale": p["churn_predicted"],
        }
        if p.get("top_factors"):
            row["facteurs"] = " | ".join(f["text"] for f in p["top_factors"])
        rows.append(row)
    table = pd.DataFrame(rows)
    return table.sort_values("probabilite_depart", ascending=False, kind="stable").reset_index(
        drop=True
    )


def risk_distribution(predictions: list[dict[str, Any]]) -> pd.DataFrame:
    """Nombre et part de clients par niveau de risque (ordre : élevé, moyen, faible)."""
    counts = pd.Series([p["risk_level"] for p in predictions]).value_counts()
    levels = sorted(RISK_ORDER, key=RISK_ORDER.get)
    out = pd.DataFrame(
        {"niveau_risque": levels, "clients": [int(counts.get(lv, 0)) for lv in levels]}
    )
    out["part"] = out["clients"] / max(len(predictions), 1)
    return out


class ApiClient:
    """Appels à l'API ChurnGuard (adresse et clé lues dans l'environnement par défaut)."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        base_url = base_url or os.getenv("API_URL", "http://127.0.0.1:8000")
        api_key = api_key if api_key is not None else os.getenv("CHURNGUARD_API_KEY", "")
        headers = {"X-API-Key": api_key} if api_key else {}
        self.http = httpx.Client(
            base_url=base_url, headers=headers, timeout=timeout, transport=transport
        )

    def _send(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Requête HTTP ; une erreur réseau (API injoignable, délai dépassé) devient ApiError."""
        try:
            return self.http.request(method, url, **kwargs)
        except httpx.HTTPError as e:
            raise ApiError(503, f"API injoignable ({type(e).__name__} : {e})") from e

    def _check(self, response: httpx.Response) -> Any:
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ApiError(response.status_code, detail)
        return response.json()

    def health(self) -> dict[str, Any]:
        return self._check(self._send("GET", "/health"))

    def model_info(self) -> dict[str, Any]:
        return self._check(self._send("GET", "/model/info"))

    def predict(self, customer: dict[str, Any]) -> dict[str, Any]:
        return self._check(self._send("POST", "/predict", json=customer))

    def predict_csv(
        self, content: bytes, include_factors: bool, batch_size: int
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Envoie un CSV par lots de `batch_size` lignes ; renvoie toutes les prédictions.

        Les numéros de ligne des erreurs sont recalés sur le fichier complet.
        """
        try:
            df = pd.read_csv(
                io.BytesIO(content), dtype=str, keep_default_na=False, encoding="utf-8-sig"
            )
        except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
            raise ApiError(400, f"CSV illisible : {e}") from e
        predictions, ignored = [], []
        for start, end in chunk_ranges(len(df), batch_size):
            part = df.iloc[start:end].to_csv(index=False).encode("utf-8")
            response = self._send(
                "POST",
                "/predict/batch/csv",
                params={"include_factors": include_factors},
                files={"file": ("lot.csv", part, "text/csv")},
            )
            if response.status_code == 422:
                detail = response.json().get("detail")
                if isinstance(detail, dict) and "lignes" in detail:
                    for entry in detail["lignes"]:
                        entry["ligne"] += start
                raise ApiError(422, detail)
            body = self._check(response)
            predictions += body["predictions"]
            ignored = body["ignored_columns"]
        return predictions, ignored

"""API FastAPI : prédiction du risque de départ et facteurs explicatifs (SHAP).

Lancement : `python tasks.py api` puis http://127.0.0.1:8000/docs (Swagger UI).
Le modèle est chargé une seule fois au démarrage ; son chemin peut être fourni par la
variable d'environnement CHURNGUARD_MODEL_PATH (sinon : chemin de config.yaml).
"""

from __future__ import annotations

import io
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from pydantic import ValidationError

from churnguard import __version__
from churnguard.api.schemas import (
    BatchRequest,
    BatchResponse,
    Customer,
    Health,
    Prediction,
    SinglePrediction,
)
from churnguard.config import load_config, resolve_path
from churnguard.data import clean_features
from churnguard.explain import ChurnExplainer
from churnguard.predict import RISK_LEVELS, check_risk_bounds, load_model, predict

MAX_CSV_BYTES = 5 * 1024 * 1024
MAX_CSV_ERRORS = 20
IncludeFactors = Annotated[
    bool,
    Query(description="Calculer les 5 facteurs principaux (SHAP) de chaque client (plus lent)"),
]
CsvFile = Annotated[UploadFile, File(description="CSV au format brut du fichier Telco")]


class ChurnService:
    """Modèle, seuil et explicateur chargés une fois, partagés par toutes les requêtes."""

    def __init__(
        self,
        artifact: dict[str, Any],
        medium_from: float,
        max_batch_size: int,
        max_batch_size_with_factors: int,
    ) -> None:
        check_risk_bounds(artifact["threshold"], medium_from)
        self.artifact = artifact
        self.medium_from = medium_from
        self.max_batch_size = max_batch_size
        self.max_batch_size_with_factors = max_batch_size_with_factors
        self.explainer = ChurnExplainer(artifact)
        self.features = artifact["numeric_features"] + artifact["categorical_features"]

    @property
    def threshold(self) -> float:
        return float(self.artifact["threshold"])

    def predict_customers(
        self, customers: list[Customer], include_factors: bool
    ) -> list[Prediction]:
        """Même préparation qu'à l'entraînement (`clean_features`), puis prédiction."""
        raw = pd.DataFrame([c.model_dump() for c in customers])
        X = clean_features(raw)
        result = predict(self.artifact, X, self.medium_from)
        factors = (
            self.explainer.top_factors_batch(X[self.features])
            if include_factors
            else [None] * len(X)
        )
        return [
            Prediction(
                customerID=c.customerID,
                churn_probability=float(r.churn_probability),
                churn_predicted=bool(r.churn_predicted),
                risk_level=r.risk_level,
                top_factors=f,
            )
            for c, r, f in zip(customers, result.itertuples(), factors, strict=True)
        ]

    def batch_response(
        self,
        customers: list[Customer],
        include_factors: bool,
        ignored_columns: list[str] | None = None,
    ) -> BatchResponse:
        # SHAP coûte environ 9 ms par client : limite plus basse quand les facteurs sont demandés.
        limit = self.max_batch_size_with_factors if include_factors else self.max_batch_size
        if len(customers) > limit:
            hint = " avec include_factors=true" if include_factors else ""
            raise HTTPException(
                status_code=413,
                detail=f"Trop de clients : {len(customers)} (maximum {limit}{hint}).",
            )
        predictions = self.predict_customers(customers, include_factors)
        counts = {level: 0 for level in RISK_LEVELS}
        for p in predictions:
            counts[p.risk_level] += 1
        return BatchResponse(
            n_customers=len(predictions),
            threshold=self.threshold,
            include_factors=include_factors,
            risk_counts=counts,
            ignored_columns=ignored_columns or [],
            predictions=predictions,
        )


def parse_csv(content: bytes) -> tuple[list[Customer], list[str]]:
    """Lit un CSV au format brut ; renvoie les clients validés et les colonnes ignorées.

    Erreur 422 si le fichier est illisible, s'il manque des colonnes, ou si des lignes sont
    invalides (avec le numéro de ligne dans le fichier, l'en-tête étant la ligne 1).
    """
    try:
        df = pd.read_csv(
            io.BytesIO(content), dtype=str, keep_default_na=False, encoding="utf-8-sig"
        )
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise HTTPException(status_code=422, detail=f"CSV illisible : {e}") from e

    expected = list(Customer.model_fields)
    required = [c for c in expected if c != "customerID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Colonnes manquantes : {', '.join(missing)}")
    if df.empty:
        raise HTTPException(status_code=422, detail="Le fichier ne contient aucun client.")
    ignored = [c for c in df.columns if c not in expected]

    customers, errors = [], []
    for i, row in enumerate(df[[c for c in expected if c in df.columns]].to_dict("records")):
        record = {k: (v.strip() or None) for k, v in row.items()}
        try:
            customers.append(Customer.model_validate(record))
        except ValidationError as e:
            msgs = [
                f"{'.'.join(map(str, err['loc'])) or 'client'} : {err['msg']}" for err in e.errors()
            ]
            errors.append({"ligne": i + 2, "erreurs": msgs})
    if errors:
        raise HTTPException(
            status_code=422,
            detail={
                "message": f"{len(errors)} ligne(s) invalide(s) sur {len(df)}.",
                "lignes": errors[:MAX_CSV_ERRORS],
            },
        )
    return customers, ignored


def create_app(model_path: str | Path | None = None) -> FastAPI:
    """Crée l'application ; `model_path` permet aux tests d'injecter un modèle d'essai."""
    cfg = load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        path = (
            model_path
            or os.getenv("CHURNGUARD_MODEL_PATH")
            or resolve_path(cfg["model"]["artifact"])
        )
        app.state.service = ChurnService(
            load_model(path),
            medium_from=cfg["risk_levels"]["medium_from"],
            max_batch_size=cfg["api"]["max_batch_size"],
            max_batch_size_with_factors=cfg["api"]["max_batch_size_with_factors"],
        )
        yield

    app = FastAPI(
        title="ChurnGuard API",
        version=__version__,
        description=(
            "Prédiction du risque de départ d'un client télécom, niveau de risque "
            "(faible / moyen / élevé) et principaux facteurs explicatifs (SHAP). "
            "« élevé » correspond exactement aux clients signalés par le seuil du modèle."
        ),
        lifespan=lifespan,
    )

    def service(request: Request) -> ChurnService:
        return request.app.state.service

    @app.get("/health", response_model=Health, tags=["état"])
    def health(request: Request) -> Health:
        """État du service et version du modèle chargé."""
        s = service(request)
        return Health(
            status="ok",
            model_name=s.artifact.get("model_name", "inconnu"),
            model_version=str(s.artifact.get("version", "inconnue")),
            threshold=s.threshold,
            risk_medium_from=s.medium_from,
            max_batch_size=s.max_batch_size,
            max_batch_size_with_factors=s.max_batch_size_with_factors,
        )

    @app.post("/predict", response_model=SinglePrediction, tags=["prédiction"])
    def predict_one(customer: Customer, request: Request) -> SinglePrediction:
        """Un client : probabilité, décision, niveau de risque et 5 facteurs principaux."""
        s = service(request)
        p = s.predict_customers([customer], include_factors=True)[0]
        return SinglePrediction(**p.model_dump(), threshold=s.threshold)

    @app.post("/predict/batch", response_model=BatchResponse, tags=["prédiction"])
    def predict_batch(
        batch: BatchRequest, request: Request, include_factors: IncludeFactors = False
    ) -> BatchResponse:
        """Plusieurs clients (JSON) ; résultats dans le même ordre que l'entrée."""
        return service(request).batch_response(batch.customers, include_factors)

    @app.post("/predict/batch/csv", response_model=BatchResponse, tags=["prédiction"])
    def predict_batch_csv(
        request: Request,
        file: CsvFile,
        include_factors: IncludeFactors = False,
    ) -> BatchResponse:
        """Plusieurs clients (fichier CSV) ; les colonnes en trop (ex. Churn) sont ignorées."""
        content = file.file.read(MAX_CSV_BYTES + 1)
        if len(content) > MAX_CSV_BYTES:
            raise HTTPException(status_code=413, detail="Fichier trop volumineux (max 5 Mo).")
        customers, ignored = parse_csv(content)
        return service(request).batch_response(customers, include_factors, ignored)

    return app


app = create_app()

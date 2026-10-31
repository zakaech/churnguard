"""Schémas Pydantic de l'API : format d'entrée (format brut du fichier) et réponses."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

YesNo = Literal["Yes", "No"]
InternetOption = Literal["Yes", "No", "No internet service"]
INTERNET_OPTIONS = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)
RiskLevel = Literal["faible", "moyen", "élevé"]

# Client fictif (pas une ligne du jeu de données), utilisé comme exemple dans Swagger UI.
EXAMPLE_CUSTOMER = {
    "customerID": "FICTIF-A",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 171.0,
}
EXAMPLE_CUSTOMER_2 = EXAMPLE_CUSTOMER | {
    "customerID": "FICTIF-C",
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 60,
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Credit card (automatic)",
    "MonthlyCharges": 75.0,
    "TotalCharges": 4500.0,
}


class Customer(BaseModel):
    """Un client, au format brut du fichier Telco (mêmes colonnes, mêmes valeurs)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_CUSTOMER]})

    customerID: str | None = Field(default=None, max_length=64, description="Facultatif")
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1] = Field(description="1 si le client est senior, 0 sinon")
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=1200, description="Ancienneté en mois")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetOption
    OnlineBackup: InternetOption
    DeviceProtection: InternetOption
    TechSupport: InternetOption
    StreamingTV: InternetOption
    StreamingMovies: InternetOption
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(gt=0, le=10_000, allow_inf_nan=False)
    TotalCharges: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        description="Peut être vide uniquement pour un client d'ancienneté 0",
    )

    @field_validator("SeniorCitizen", mode="before")
    @classmethod
    def senior_as_int(cls, value: object) -> object:
        """Accepte 0/1 (et "0"/"1", issus d'un CSV) ; refuse les booléens."""
        if isinstance(value, bool):
            raise ValueError("SeniorCitizen doit valoir 0 ou 1 (pas true/false).")
        if isinstance(value, str) and value.strip() in ("0", "1"):
            return int(value.strip())
        return value

    @model_validator(mode="after")
    def check_consistency(self) -> Customer:
        """Règles métier : mêmes règles que le nettoyage et que les données d'entraînement."""
        if self.TotalCharges is None and self.tenure > 0:
            raise ValueError("TotalCharges ne peut être vide que pour un client d'ancienneté 0.")
        no_internet = [o for o in INTERNET_OPTIONS if getattr(self, o) == "No internet service"]
        if self.InternetService == "No" and len(no_internet) != len(INTERNET_OPTIONS):
            raise ValueError(
                "Sans internet (InternetService = 'No'), toutes les options internet doivent "
                "valoir 'No internet service'."
            )
        if self.InternetService != "No" and no_internet:
            raise ValueError(
                f"Avec internet, les options ne peuvent pas valoir 'No internet service' "
                f"({', '.join(no_internet)})."
            )
        if (self.PhoneService == "No") != (self.MultipleLines == "No phone service"):
            raise ValueError(
                "MultipleLines doit valoir 'No phone service' si et seulement si "
                "PhoneService = 'No'."
            )
        return self


class BatchRequest(BaseModel):
    """Plusieurs clients à évaluer."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"customers": [EXAMPLE_CUSTOMER, EXAMPLE_CUSTOMER_2]}]},
    )

    customers: list[Customer] = Field(min_length=1)


class Factor(BaseModel):
    """Un facteur de risque, lisible par un non-technicien."""

    feature: str
    label: str
    value: str | int | float | None
    value_label: str
    impact_points: float = Field(description="Impact sur la probabilité, en points (%)")
    direction: Literal["hausse", "baisse", "neutre"]
    text: str


class Prediction(BaseModel):
    customerID: str | None
    churn_probability: float = Field(ge=0, le=1)
    churn_predicted: bool = Field(description="Signalé : probabilité >= seuil")
    risk_level: RiskLevel
    top_factors: list[Factor] | None = Field(
        default=None, description="5 facteurs principaux (SHAP), si demandés"
    )


class SinglePrediction(Prediction):
    threshold: float


class BatchResponse(BaseModel):
    n_customers: int
    threshold: float
    include_factors: bool
    risk_counts: dict[str, int]
    ignored_columns: list[str] = Field(
        default_factory=list, description="Colonnes du CSV non utilisées (ex. Churn)"
    )
    predictions: list[Prediction]


class ImportanceItem(BaseModel):
    feature: str
    label: str
    mean_abs_points: float = Field(description="Impact moyen absolu (points de probabilité)")


class ModelInfo(BaseModel):
    """Informations pour la page d'analyse : performances et importance globale."""

    model_name: str
    model_version: str
    created_at: str | None
    threshold: float
    threshold_rule: str | None
    risk_medium_from: float
    validation_metrics: dict[str, Any] | None
    test_metrics: dict[str, Any] | None
    global_importance: list[ImportanceItem] | None


class Health(BaseModel):
    status: Literal["ok"]
    model_name: str
    model_version: str
    threshold: float
    risk_medium_from: float
    max_batch_size: int
    max_batch_size_with_factors: int

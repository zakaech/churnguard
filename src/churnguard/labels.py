"""Libellés français des variables et de leurs valeurs, sans dépendance lourde.

Partagé par les explications (SHAP) et par l'interface Streamlit, qui ne doit pas importer
le modèle ni ses bibliothèques.
"""

from __future__ import annotations

import math
from typing import Any

# Libellés lisibles par un non-technicien.
FEATURE_LABELS: dict[str, str] = {
    "tenure": "Ancienneté",
    "MonthlyCharges": "Facture mensuelle",
    "TotalCharges": "Total facturé depuis le début",
    "gender": "Genre",
    "SeniorCitizen": "Client senior",
    "Partner": "Vit en couple",
    "Dependents": "Personnes à charge",
    "PhoneService": "Ligne téléphonique",
    "MultipleLines": "Plusieurs lignes",
    "InternetService": "Type d'internet",
    "OnlineSecurity": "Option sécurité en ligne",
    "OnlineBackup": "Option sauvegarde en ligne",
    "DeviceProtection": "Option protection de l'appareil",
    "TechSupport": "Option support technique",
    "StreamingTV": "Option TV en streaming",
    "StreamingMovies": "Option films en streaming",
    "Contract": "Contrat",
    "PaperlessBilling": "Facture dématérialisée",
    "PaymentMethod": "Moyen de paiement",
}

COMMON_VALUES: dict[str, str] = {
    "Yes": "oui",
    "No": "non",
    "No internet service": "pas d'internet",
    "No phone service": "pas de ligne téléphonique",
}

VALUE_LABELS: dict[str, dict[str, str]] = {
    "gender": {"Male": "homme", "Female": "femme"},
    "InternetService": {"DSL": "ADSL", "Fiber optic": "fibre optique", "No": "aucun"},
    "Contract": {"Month-to-month": "au mois", "One year": "un an", "Two year": "deux ans"},
    "PaymentMethod": {
        "Electronic check": "chèque électronique",
        "Mailed check": "chèque papier",
        "Bank transfer (automatic)": "virement automatique",
        "Credit card (automatic)": "carte bancaire (automatique)",
    },
}


def format_number(x: float, decimals: int = 2) -> str:
    """Nombre au format français (virgule décimale, espace pour les milliers)."""
    return f"{x:,.{decimals}f}".replace(",", " ").replace(".", ",")


def value_label(feature: str, value: Any) -> str:
    """Valeur brute d'une variable, rendue lisible (« au mois », « 12 mois », « 70,35 »)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "non renseigné"
    if feature == "tenure":
        return f"{int(value)} mois"
    if feature in ("MonthlyCharges", "TotalCharges"):
        return format_number(float(value))
    value = str(value)
    return VALUE_LABELS.get(feature, {}).get(value, COMMON_VALUES.get(value, value))

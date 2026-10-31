"""Génère un petit échantillon SYNTHÉTIQUE au format du jeu Telco Customer Churn.

Aucune ligne ne provient des vraies données : seules les colonnes, les types et les
modalités possibles sont reproduits. Cet échantillon est versionné (tests/data/sample.csv)
pour que les tests et la CI ne dépendent ni de Kaggle ni de la licence du jeu réel.

Particularités reproduites volontairement :
- TotalCharges stocké en texte, avec quelques valeurs vides (" ") pour tenure = 0 ;
- cohérence des services ("No internet service", "No phone service") ;
- Churn en Yes/No, généré par une règle simple pour qu'un modèle puisse apprendre un signal.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

INTERNET_ADDONS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def generate(n_rows: int, seed: int, n_blank_total: int = 3) -> pd.DataFrame:
    """Construit `n_rows` clients synthétiques reproductibles pour une graine donnée."""
    rng = np.random.default_rng(seed)

    def choice(values: list[str], p: list[float] | None = None) -> np.ndarray:
        return rng.choice(values, size=n_rows, p=p)

    df = pd.DataFrame(
        {
            "customerID": [f"SYN-{i:05d}" for i in range(n_rows)],
            "gender": choice(["Male", "Female"]),
            "SeniorCitizen": rng.choice([0, 1], size=n_rows, p=[0.8, 0.2]),
            "Partner": choice(["Yes", "No"]),
            "Dependents": choice(["Yes", "No"], p=[0.3, 0.7]),
            "tenure": rng.integers(1, 73, size=n_rows),
            "PhoneService": choice(["Yes", "No"], p=[0.9, 0.1]),
            "InternetService": choice(["DSL", "Fiber optic", "No"], p=[0.35, 0.45, 0.2]),
            "Contract": choice(["Month-to-month", "One year", "Two year"], p=[0.55, 0.2, 0.25]),
            "PaperlessBilling": choice(["Yes", "No"], p=[0.6, 0.4]),
            "PaymentMethod": choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ]
            ),
        }
    )

    df["MultipleLines"] = np.where(
        df["PhoneService"] == "No", "No phone service", choice(["Yes", "No"])
    )
    for col in INTERNET_ADDONS:
        df[col] = np.where(
            df["InternetService"] == "No", "No internet service", choice(["Yes", "No"])
        )

    base = np.select(
        [df["InternetService"] == "Fiber optic", df["InternetService"] == "DSL"],
        [75.0, 50.0],
        default=20.0,
    )
    monthly = base + 5 * (df[INTERNET_ADDONS] == "Yes").sum(axis=1) + rng.normal(0, 3, n_rows)
    df["MonthlyCharges"] = np.round(np.clip(monthly, 18.0, 120.0), 2)

    total = df["tenure"] * df["MonthlyCharges"] * rng.uniform(0.95, 1.05, n_rows)
    df["TotalCharges"] = [f"{v:.2f}" for v in total]

    # Quelques nouveaux clients (tenure = 0) avec TotalCharges vide, comme dans le jeu réel.
    blank_idx = rng.choice(n_rows, size=n_blank_total, replace=False)
    df.loc[blank_idx, "tenure"] = 0
    df.loc[blank_idx, "TotalCharges"] = " "

    # Règle de départ synthétique (logistique) : contrat mensuel, faible ancienneté,
    # fibre et paiement par chèque électronique augmentent le risque.
    logit = (
        -1.5
        + 1.5 * (df["Contract"] == "Month-to-month")
        - 0.03 * df["tenure"]
        + 0.8 * (df["InternetService"] == "Fiber optic")
        + 0.5 * (df["PaymentMethod"] == "Electronic check")
    )
    proba = 1 / (1 + np.exp(-logit))
    df["Churn"] = np.where(rng.uniform(size=n_rows) < proba, "Yes", "No")

    columns = [
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", *INTERNET_ADDONS,
        "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
        "Churn",
    ]  # fmt: skip
    return df[columns]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("tests/data/sample.csv"))
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} lignes synthétiques écrites dans {args.out}")
    print(f"Taux de départ : {(df['Churn'] == 'Yes').mean():.2%}")


if __name__ == "__main__":
    main()

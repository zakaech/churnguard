"""Explicabilité du modèle avec SHAP : contributions par variable et facteurs lisibles.

Les valeurs SHAP sont calculées sur les variables transformées (one-hot, normalisées), puis
regroupées par variable d'origine : la contribution de « Contrat » est la somme des
contributions de ses colonnes one-hot. L'additivité est conservée :
    valeur de base + somme des contributions = probabilité de départ prédite.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

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
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "non renseigné"
    if feature == "tenure":
        return f"{int(value)} mois"
    if feature in ("MonthlyCharges", "TotalCharges"):
        return format_number(float(value))
    value = str(value)
    return VALUE_LABELS.get(feature, {}).get(value, COMMON_VALUES.get(value, value))


class ChurnExplainer:
    """Explique les prédictions d'un artefact ChurnGuard (pipeline + seuil)."""

    def __init__(self, artifact: dict[str, Any]) -> None:
        pipeline = artifact["pipeline"]
        self.preprocess = pipeline.named_steps["preprocess"]
        self.model = pipeline.named_steps["model"]
        if not isinstance(self.model, RandomForestClassifier):
            # Pour un Random Forest scikit-learn, TreeExplainer donne des contributions
            # directement en probabilité ; ce n'est pas le cas pour XGBoost (log-odds).
            raise NotImplementedError(
                f"Explications en points de probabilité non prises en charge pour "
                f"{type(self.model).__name__}."
            )
        self.features: list[str] = artifact["numeric_features"] + artifact["categorical_features"]
        self.explainer = shap.TreeExplainer(self.model)
        self.base_value = float(np.atleast_1d(self.explainer.expected_value)[-1])
        self.groups = self._feature_groups()

    def _feature_groups(self) -> dict[str, list[int]]:
        """Indices des colonnes transformées appartenant à chaque variable d'origine."""
        groups: dict[str, list[int]] = {}
        start = 0
        for name, transformer, columns in self.preprocess.transformers_:
            if name == "remainder" or transformer == "drop":
                continue
            if name == "num":
                widths = [1] * len(columns)
            else:
                onehot = transformer.named_steps["onehot"]
                drop = onehot.drop_idx_ if onehot.drop_idx_ is not None else [None] * len(columns)
                widths = [
                    len(cats) - (d is not None)
                    for cats, d in zip(onehot.categories_, drop, strict=True)
                ]
            for col, width in zip(columns, widths, strict=True):
                groups[col] = list(range(start, start + width))
                start += width
        n_out = len(self.preprocess.get_feature_names_out())
        if start != n_out:
            raise RuntimeError(f"Regroupement incohérent : {start} colonnes sur {n_out}.")
        return groups

    def transformed_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Valeurs SHAP (classe « départ ») sur les colonnes transformées."""
        values = self.explainer.shap_values(self.preprocess.transform(X[self.features]))
        values = values[1] if isinstance(values, list) else np.asarray(values)
        return values[..., 1] if values.ndim == 3 else values

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Contribution de chaque variable d'origine à la probabilité, une ligne par client."""
        values = self.transformed_shap_values(X)
        return pd.DataFrame(
            {feature: values[:, idx].sum(axis=1) for feature, idx in self.groups.items()},
            index=X.index,
        )[self.features]

    def top_factors(self, client: pd.DataFrame | pd.Series, k: int = 5) -> list[dict[str, Any]]:
        """Les `k` facteurs les plus importants (en valeur absolue, dans les deux sens)."""
        row = client.to_frame().T if isinstance(client, pd.Series) else client
        if len(row) != 1:
            raise ValueError("top_factors attend un seul client.")
        contrib = self.contributions(row).iloc[0]
        order = contrib.abs().sort_values(ascending=False, kind="stable").index[:k]
        return [
            self._factor(feature, row.iloc[0][feature], float(contrib[feature]))
            for feature in order
        ]

    def _factor(self, feature: str, value: Any, impact: float) -> dict[str, Any]:
        points = 100 * impact
        if impact > 0:
            direction, verb = "hausse", "augmente le risque"
        elif impact < 0:
            direction, verb = "baisse", "diminue le risque"
        else:
            direction, verb = "neutre", "sans effet sur le risque"
        label, shown = FEATURE_LABELS.get(feature, feature), value_label(feature, value)
        sign = "+" if points > 0 else "−" if points < 0 else ""
        return {
            "feature": feature,
            "label": label,
            "value": None if pd.isna(value) else value,
            "value_label": shown,
            "impact": impact,
            "impact_points": round(points, 1),
            "direction": direction,
            "text": f"{label} : {shown} — {verb} ({sign}{format_number(abs(points), 1)} points)",
        }

    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Importance globale : moyenne des |contributions| par variable, triée."""
        mean_abs = self.contributions(X).abs().mean().sort_values(ascending=False)
        return pd.DataFrame(
            {
                "feature": mean_abs.index,
                "label": [FEATURE_LABELS.get(f, f) for f in mean_abs.index],
                "mean_abs_impact": mean_abs.to_numpy(),
                "mean_abs_points": 100 * mean_abs.to_numpy(),
            }
        )


def plot_global_importance(importance: pd.DataFrame, path) -> None:
    """Barres horizontales : impact moyen absolu (points de probabilité) par variable."""
    from pathlib import Path

    import matplotlib.pyplot as plt

    data = importance.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.barh(data["label"], data["mean_abs_points"], color="#2a78d6", height=0.65)
    for y, v in enumerate(data["mean_abs_points"]):
        ax.text(v + 0.1, y, format_number(v, 1), va="center", fontsize=9)
    ax.set_xlabel("impact moyen sur la probabilité de départ (points, valeur absolue)")
    ax.set_title(
        "Importance globale des variables (SHAP, validation)", loc="left", fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --- Ligne de commande ---------------------------------------------------------------------


def run_global(cfg: dict[str, Any]) -> pd.DataFrame:
    """Importance globale sur la validation, enregistrée dans reports/."""
    from churnguard.config import resolve_path
    from churnguard.data import get_splits
    from churnguard.predict import load_model

    explainer = ChurnExplainer(load_model(resolve_path(cfg["model"]["artifact"])))
    importance = explainer.global_importance(get_splits(cfg)["val"])
    reports = resolve_path(cfg["paths"]["reports_dir"])
    importance.to_csv(reports / "shap_global_importance.csv", index=False)
    plot_global_importance(importance, reports / "figures" / "shap_global_importance.png")
    return importance


def run_customer(cfg: dict[str, Any], customer_id: str, split: str, k: int = 5) -> str:
    """Explication lisible d'un client réel (affichée seulement, rien n'est enregistré)."""
    from churnguard.config import resolve_path
    from churnguard.data import get_splits
    from churnguard.predict import load_model, predict

    artifact = load_model(resolve_path(cfg["model"]["artifact"]))
    data = get_splits(cfg)[split]
    match = data[data[cfg["columns"]["id"]] == customer_id]
    if match.empty:
        raise KeyError(f"Client {customer_id} introuvable dans le jeu « {split} ».")
    result = predict(artifact, match).iloc[0]
    factors = ChurnExplainer(artifact).top_factors(match, k=k)
    decision = "À RISQUE (signalé)" if result["churn_predicted"] else "non signalé"
    lines = [
        f"Client {customer_id} ({split})",
        f"Probabilité de départ : {format_number(100 * result['churn_probability'], 1)} % "
        f"(seuil {format_number(100 * artifact['threshold'], 1)} %) -> {decision}",
        f"Principaux facteurs ({k}, par impact décroissant) :",
        *[f"  {i}. {f['text']}" for i, f in enumerate(factors, 1)],
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from churnguard.config import load_config

    parser = argparse.ArgumentParser(description="Explications SHAP du modèle ChurnGuard")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("global", help="Importance globale sur la validation")
    customer = sub.add_parser("customer", help="Facteurs de risque d'un client")
    customer.add_argument("--customer-id", required=True)
    customer.add_argument("--split", choices=["train", "val", "test"], default="test")
    customer.add_argument("-k", type=int, default=5)
    args = parser.parse_args(argv)

    cfg = load_config()
    if args.command == "global":
        importance = run_global(cfg)
        print("Importance globale (validation, points de probabilité) :")
        for _, r in importance.iterrows():
            print(f"  {r['label']:<34} {format_number(r['mean_abs_points'], 1):>5}")
        return 0
    try:
        print(run_customer(cfg, args.customer_id, args.split, args.k))
    except KeyError as e:
        print(e.args[0], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Métriques, choix du seuil, analyse de coût et figures d'évaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray | pd.Series, y_proba: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    """Calcule les métriques à partir des probabilités de départ (classe 1).

    - ROC-AUC et PR-AUC dépendent seulement du classement des probabilités (pas du seuil).
    - Précision, rappel, F1, exactitude et matrice de confusion dépendent du seuil.
    La PR-AUC est estimée par l'average precision de scikit-learn.
    """
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def metrics_table(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Met les métriques de plusieurs modèles côte à côte (une ligne par modèle)."""
    rows = {
        name: {
            **{k: v for k, v in m.items() if k != "confusion_matrix"},
            **m["confusion_matrix"],
        }
        for name, m in results.items()
    }
    table = pd.DataFrame(rows).T
    counts = ["tn", "fp", "fn", "tp"]
    return table.astype(
        {c: float for c in table.columns if c not in counts} | dict.fromkeys(counts, int)
    )


def select_threshold(
    y_true: np.ndarray | pd.Series, y_proba: np.ndarray, target_recall: float
) -> dict[str, float]:
    """Choisit le seuil qui maximise la précision parmi ceux qui atteignent le rappel visé.

    Règle de décision : un client est signalé si sa probabilité est >= au seuil.
    En cas d'égalité de précision, on garde le seuil le plus élevé (moins d'alertes).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # Le dernier point de la courbe (rappel 0) n'a pas de seuil associé.
    precision, recall = precision[:-1], recall[:-1]
    ok = recall >= target_recall
    if not ok.any():
        raise ValueError(f"Aucun seuil n'atteint un rappel de {target_recall:.0%}.")
    best = np.flatnonzero(ok & (precision == precision[ok].max()))
    idx = best[np.argmax(thresholds[best])]
    return {
        "threshold": float(thresholds[idx]),
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
    }


def precision_at_recall(
    y_true: np.ndarray | pd.Series, y_proba: np.ndarray, target_recall: float
) -> float:
    """Meilleure précision atteignable avec un rappel >= `target_recall`."""
    return select_threshold(y_true, y_proba, target_recall)["precision"]


def cost_parameters(cfg: dict[str, Any], train: pd.DataFrame) -> dict[str, float]:
    """Convertit les hypothèses de coût (en mois) en montants via la facture médiane du train."""
    ca = cfg["cost_analysis"]
    monthly = float(train["MonthlyCharges"].median())
    return {
        "median_monthly_charges": monthly,
        "lost_customer_cost": ca["lost_customer_months"] * monthly,
        "offer_cost": ca["offer_months"] * monthly,
        "offer_success_rate": ca["offer_success_rate"],
    }


def business_cost(
    confusion: dict[str, int],
    lost_customer_cost: float,
    offer_cost: float,
    offer_success_rate: float,
) -> float:
    """Coût total d'une politique de ciblage (hypothèses illustratives).

    coût = FN × coût_client_perdu                       (départs non détectés)
         + TP × (1 − taux_succès) × coût_client_perdu   (détectés mais perdus malgré l'offre)
         + (TP + FP) × coût_offre                       (offres envoyées)
    """
    tp, fp, fn = confusion["tp"], confusion["fp"], confusion["fn"]
    return (
        fn * lost_customer_cost
        + tp * (1 - offer_success_rate) * lost_customer_cost
        + (tp + fp) * offer_cost
    )


def _cost_at(y_true, y_proba, params: dict[str, float], threshold: float) -> dict[str, Any]:
    cm = compute_metrics(y_true, y_proba, threshold=threshold)["confusion_matrix"]
    cost = business_cost(
        cm, params["lost_customer_cost"], params["offer_cost"], params["offer_success_rate"]
    )
    return {"threshold": threshold, **cm, "cost": cost}


def cost_curve(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    params: dict[str, float],
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Coût total pour une grille de seuils (par défaut 0,01 à 0,99)."""
    if thresholds is None:
        thresholds = np.round(np.linspace(0.01, 0.99, 99), 2)
    return pd.DataFrame([_cost_at(y_true, y_proba, params, float(t)) for t in thresholds])


def cost_analysis(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    params: dict[str, float],
    chosen_threshold: float,
) -> dict[str, Any]:
    """Compare le coût au seuil retenu, au seuil 0,5, au seuil de coût minimal et sans action."""
    curve = cost_curve(y_true, y_proba, params)
    best = curve.loc[curve["cost"].idxmin(), "threshold"]
    return {
        "hypotheses": params,
        "no_action_cost": int(np.asarray(y_true).sum()) * params["lost_customer_cost"],
        "chosen_threshold": _cost_at(y_true, y_proba, params, chosen_threshold),
        "threshold_0_5": _cost_at(y_true, y_proba, params, 0.5),
        "min_cost_threshold": _cost_at(y_true, y_proba, params, float(best)),
    }


# --- Figures -------------------------------------------------------------------------------

BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#52514e"


def plot_threshold_analysis(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    chosen: dict[str, float],
    target_recall: float,
    path: str | Path,
) -> None:
    """Courbe précision-rappel et précision/rappel selon le seuil, avec le seuil retenu."""
    import matplotlib.pyplot as plt

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    ax1.plot(recall, precision, color=BLUE, lw=2)
    ax1.axvline(target_recall, color=MUTED, lw=1, ls="--")
    ax1.scatter(
        [chosen["recall"]],
        [chosen["precision"]],
        s=60,
        color=ORANGE,
        zorder=3,
        edgecolor="white",
        linewidth=2,
    )
    ax1.annotate(
        f"seuil {chosen['threshold']:.3f}\nP = {chosen['precision']:.2f}, "
        f"R = {chosen['recall']:.2f}",
        (chosen["recall"], chosen["precision"]),
        xytext=(-150, -70),
        textcoords="offset points",
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": MUTED},
    )
    ax1.set(xlabel="rappel", ylabel="précision", xlim=(0, 1.02), ylim=(0, 1.02))
    ax1.set_title("Courbe précision-rappel (validation)", loc="left", fontweight="bold")

    ax2.plot(thresholds, precision[:-1], color=BLUE, lw=2, label="précision")
    ax2.plot(thresholds, recall[:-1], color=ORANGE, lw=2, label="rappel")
    ax2.axhline(target_recall, color=MUTED, lw=1, ls=":")
    ax2.axvline(chosen["threshold"], color=MUTED, lw=1, ls="--")
    ax2.set(xlabel="seuil de décision", ylim=(0, 1.02))
    ax2.legend(frameon=False, loc="lower left")
    ax2.set_title("Précision et rappel selon le seuil (validation)", loc="left", fontweight="bold")

    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_cost_curve(curve: pd.DataFrame, analysis: dict[str, Any], path: str | Path) -> None:
    """Coût total (hypothèses illustratives) selon le seuil."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(curve["threshold"], curve["cost"], color=BLUE, lw=2, label="coût total")
    ax.axvline(
        analysis["chosen_threshold"]["threshold"],
        color=MUTED,
        lw=1,
        ls="--",
        label="seuil retenu (rappel cible)",
    )
    ax.axvline(
        analysis["min_cost_threshold"]["threshold"], color=MUTED, lw=1, ls=":", label="coût minimal"
    )
    ax.axhline(analysis["no_action_cost"], color=ORANGE, lw=1.5, label="aucune action")
    ax.set(xlabel="seuil de décision", ylabel="coût total (hypothèses)")
    ax.set_title(
        "Coût total selon le seuil (validation) — montants hypothétiques",
        loc="left",
        fontweight="bold",
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --- Évaluation unique sur le test ---------------------------------------------------------


def evaluate_on_test(cfg: dict[str, Any] | None = None, force: bool = False) -> dict[str, Any]:
    """Évalue le modèle final et son seuil sur le test, une seule fois.

    Refuse de recommencer si le rapport de test existe déjà (sauf `force=True`) :
    regarder le test plusieurs fois inciterait à ajuster le modèle d'après lui,
    et son estimation ne serait plus honnête.
    """
    import json

    from churnguard.config import load_config, resolve_path
    from churnguard.data import get_splits, split_xy
    from churnguard.predict import load_model, predict_proba

    cfg = cfg or load_config()
    out = resolve_path(cfg["paths"]["reports_dir"]) / "test_metrics.json"
    if out.exists() and not force:
        raise RuntimeError(
            f"Le test a déjà été évalué ({out}). Il ne doit l'être qu'une fois ; "
            "utiliser --force seulement en connaissance de cause."
        )

    artifact = load_model(resolve_path(cfg["model"]["artifact"]))
    splits = get_splits(cfg)
    X_test, y_test = split_xy(splits["test"], cfg)
    y_proba = predict_proba(artifact, X_test)
    metrics = compute_metrics(y_test, y_proba, threshold=artifact["threshold"])
    costs = cost_analysis(
        y_test, y_proba, cost_parameters(cfg, splits["train"]), artifact["threshold"]
    )
    result = {"model": artifact["model_name"], "metrics": metrics, "cost_analysis": costs}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    meta_path = resolve_path(cfg["model"]["metadata"])
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["test_metrics"] = metrics
    metadata["test_cost_analysis"] = costs
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    _log_test_to_mlflow(cfg, metadata.get("mlflow_run_id"), metrics)
    return result


def _log_test_to_mlflow(cfg: dict[str, Any], run_id: str | None, metrics: dict[str, Any]) -> None:
    if run_id is None:
        return
    import mlflow

    from churnguard.train import setup_mlflow

    setup_mlflow(cfg)
    with mlflow.start_run(run_id=run_id):
        flat = {f"test_{k}": float(v) for k, v in metrics.items() if k != "confusion_matrix"}
        flat |= {f"test_{k}": float(v) for k, v in metrics["confusion_matrix"].items()}
        mlflow.log_metrics(flat)


def main() -> None:
    import sys

    result = evaluate_on_test(force="--force" in sys.argv)
    m = result["metrics"]
    print(f"TEST (évaluation unique) — modèle {result['model']}, seuil {m['threshold']:.4f}")
    for k in ("precision", "recall", "f1", "roc_auc", "pr_auc", "accuracy"):
        print(f"  {k:<10} {m[k]:.3f}")
    print(f"  matrice de confusion : {m['confusion_matrix']}")


if __name__ == "__main__":
    main()

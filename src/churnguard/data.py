"""Chargement, nettoyage et découpage stratifié des données Telco Customer Churn."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from churnguard.config import load_config, resolve_path

SPLIT_NAMES = ("train", "val", "test")


def load_raw(path: str | Path) -> pd.DataFrame:
    """Lit le CSV brut sans aucune transformation."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Nettoie les données brutes et renvoie une copie (le DataFrame d'entrée n'est pas modifié).

    - TotalCharges : texte -> nombre. Les valeurs vides ne concernent que des clients
      d'ancienneté 0 (jamais facturés) et sont mises à 0. Toute autre valeur non numérique
      lève une erreur plutôt que d'être masquée.
    - SeniorCitizen : 0/1 -> "No"/"Yes", pour être codé comme les autres variables binaires.
    - Churn : "Yes"/"No" -> 1/0.

    Ces règles s'appliquent ligne par ligne, sans statistique apprise sur les données :
    elles ne créent donc pas de fuite et peuvent être faites avant le découpage.
    """
    cols = cfg["columns"]
    target = cols["target"]
    out = df.copy()

    total = out["TotalCharges"].astype("string").str.strip()
    blank = total.isna() | (total == "")
    if (out.loc[blank, "tenure"] != 0).any():
        raise ValueError("TotalCharges vide pour un client d'ancienneté > 0 : cas non prévu.")
    out["TotalCharges"] = pd.to_numeric(total.mask(blank, "0"), errors="raise").astype(float)

    senior = out["SeniorCitizen"]
    if not senior.isin([0, 1]).all():
        raise ValueError("SeniorCitizen contient des valeurs autres que 0/1.")
    out["SeniorCitizen"] = senior.map({0: "No", 1: "Yes"}).astype(str)

    churn = out[target]
    if not churn.isin(["Yes", "No"]).all():
        raise ValueError(f"{target} contient des valeurs autres que Yes/No.")
    out[target] = churn.map({"Yes": 1, "No": 0}).astype(int)

    if out[cols["id"]].duplicated().any():
        raise ValueError(f"{cols['id']} contient des doublons.")
    return out


def feature_columns(cfg: dict[str, Any]) -> list[str]:
    """Liste des variables explicatives (sans l'identifiant ni la cible)."""
    return cfg["columns"]["numeric"] + cfg["columns"]["categorical"]


def split_xy(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """Sépare variables explicatives X et cible y."""
    return df[feature_columns(cfg)], df[cfg["columns"]["target"]]


def split_data(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Découpe stratifié en train / val / test selon les proportions de la configuration.

    On isole d'abord le test, puis on découpe le reste en train et validation ;
    la taille de la validation est rapportée au reste pour obtenir la proportion voulue
    du total (0,2 / 0,8 = 0,25 du reste).
    """
    seed = cfg["seed"]
    test_size = cfg["split"]["test_size"]
    val_size = cfg["split"]["val_size"]
    target = cfg["columns"]["target"]

    rest, test = train_test_split(df, test_size=test_size, stratify=df[target], random_state=seed)
    train, val = train_test_split(
        rest,
        test_size=val_size / (1 - test_size),
        stratify=rest[target],
        random_state=seed,
    )
    return {"train": train, "val": val, "test": test}


def save_splits(splits: dict[str, pd.DataFrame], directory: str | Path) -> None:
    """Écrit chaque partie en CSV dans `directory`."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(directory / f"{name}.csv", index=False)


def load_splits(directory: str | Path, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Relit les parties sauvegardées par `save_splits`."""
    directory = Path(directory)
    # SeniorCitizen ("No"/"Yes") est relu comme texte, comme les autres catégorielles.
    dtypes = {col: str for col in cfg["columns"]["categorical"]}
    return {name: pd.read_csv(directory / f"{name}.csv", dtype=dtypes) for name in SPLIT_NAMES}


def make_dataset(cfg: dict[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    """Chaîne complète : brut -> nettoyage -> découpage -> data/processed/."""
    cfg = cfg or load_config()
    raw = load_raw(resolve_path(cfg["paths"]["raw_data"]))
    splits = split_data(clean(raw, cfg), cfg)
    save_splits(splits, resolve_path(cfg["paths"]["processed_dir"]))
    return splits


def get_splits(cfg: dict[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    """Charge les parties depuis data/processed/, en les créant si besoin."""
    cfg = cfg or load_config()
    directory = resolve_path(cfg["paths"]["processed_dir"])
    if not all((directory / f"{name}.csv").exists() for name in SPLIT_NAMES):
        return make_dataset(cfg)
    return load_splits(directory, cfg)


def main() -> None:
    cfg = load_config()
    splits = make_dataset(cfg)
    target = cfg["columns"]["target"]
    total = sum(len(p) for p in splits.values())
    print(f"Données nettoyées et découpées dans {cfg['paths']['processed_dir']}/")
    for name, part in splits.items():
        print(
            f"  {name:<5} : {len(part):>5} lignes ({len(part) / total:.1%}), "
            f"taux de départ = {part[target].mean():.2%}"
        )


if __name__ == "__main__":
    main()

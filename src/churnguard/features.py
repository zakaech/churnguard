"""Pipeline de prétraitement des variables."""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(cfg: dict[str, Any]) -> ColumnTransformer:
    """Prépare les variables numériques et catégorielles.

    - Numériques : imputation par la médiane, puis normalisation.
    - Catégorielles : imputation par la modalité la plus fréquente, puis one-hot.
      `drop="if_binary"` garde une seule colonne pour les variables à deux modalités ;
      `handle_unknown="ignore"` code une modalité inconnue par des zéros au lieu d'échouer.

    Les données nettoyées n'ont pas de valeurs manquantes : l'imputation protège
    l'utilisation en production (API). Toutes ces statistiques sont apprises sur les
    données passées à `fit` (le train) car le ColumnTransformer est dans un Pipeline.
    """
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, cfg["columns"]["numeric"]),
            ("cat", categorical, cfg["columns"]["categorical"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

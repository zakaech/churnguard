"""Pipeline de prétraitement des variables (version minimale, complétée en phase 2)."""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(cfg: dict[str, Any]) -> ColumnTransformer:
    """Normalise les variables numériques et encode les catégorielles en one-hot.

    Le ColumnTransformer est placé dans un Pipeline : ses statistiques (moyennes,
    écarts-types, modalités) sont apprises uniquement sur les données passées à `fit`,
    c'est-à-dire le train. `handle_unknown="ignore"` évite une erreur si une modalité
    absente du train apparaît plus tard.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), cfg["columns"]["numeric"]),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cfg["columns"]["categorical"],
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

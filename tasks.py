"""Commandes du projet, multiplateforme (remplace un Makefile).

Usage : python tasks.py <commande>
Lancer avec l'interpréteur de l'environnement virtuel activé.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).resolve().parent

COMMANDS: dict[str, tuple[str, list[list[str]]]] = {
    "download": (
        "Télécharge les données Kaggle vers data/raw/",
        [[PY, "scripts/download_data.py"]],
    ),
    "data": (
        "Nettoie et découpe les données vers data/processed/",
        [[PY, "-m", "churnguard.data"]],
    ),
    "notebook": (
        "Ré-exécute les notebooks (exploration, résultats) : sorties régénérées",
        [
            [PY, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", nb]
            for nb in ("notebooks/01_exploration.ipynb", "notebooks/02_resultats.ipynb")
        ],
    ),
    "train-baseline": (
        "Entraîne les baselines et affiche leurs métriques sur la validation",
        [[PY, "-m", "churnguard.train", "--baseline"]],
    ),
    "train": (
        "Compare, règle, choisit modèle et seuil, sauvegarde (refusé si le test a été évalué)",
        [[PY, "-m", "churnguard.train"]],
    ),
    "evaluate-test": (
        "Évalue UNE SEULE FOIS le modèle final sur le test (--force pour forcer)",
        [[PY, "-m", "churnguard.evaluate"]],
    ),
    "explain-global": (
        "Importance globale SHAP sur la validation (reports/)",
        [[PY, "-m", "churnguard.explain", "global"]],
    ),
    "explain": (
        "Facteurs de risque d'un client : --customer-id ID [--split test|val|train]",
        [[PY, "-m", "churnguard.explain", "customer"]],
    ),
    "mlflow-ui": (
        "Lance l'interface MLflow sur http://127.0.0.1:5000",
        [[PY, "-m", "mlflow", "ui", "--backend-store-uri", "sqlite:///mlflow.db"]],
    ),
    "lint": ("Vérifie le style avec ruff", [[PY, "-m", "ruff", "check", "."]]),
    "format": (
        "Formate le code avec ruff",
        [[PY, "-m", "ruff", "format", "."], [PY, "-m", "ruff", "check", "--fix", "."]],
    ),
    "test": ("Lance les tests pytest", [[PY, "-m", "pytest"]]),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Commandes ChurnGuard")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (help_text, _) in COMMANDS.items():
        sub.add_parser(name, help=help_text)
    # Les options en plus (ex. --force) sont transmises à la dernière commande.
    args, extra = parser.parse_known_args()

    commands = COMMANDS[args.command][1]
    for i, cmd in enumerate(commands):
        if i == len(commands) - 1:
            cmd = cmd + extra
        print(f"$ {' '.join(cmd)}", flush=True)
        code = subprocess.call(cmd, cwd=ROOT, env={**os.environ, "PYTHONUTF8": "1"})
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())

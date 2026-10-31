"""Commandes du projet, multiplateforme (remplace un Makefile).

Usage : python tasks.py <commande>
Lancer avec l'interpréteur de l'environnement virtuel activé.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

PY = sys.executable

COMMANDS: dict[str, tuple[str, list[list[str]]]] = {
    "download": (
        "Télécharge les données Kaggle vers data/raw/",
        [[PY, "scripts/download_data.py"]],
    ),
    "data": (
        "Nettoie et découpe les données vers data/processed/",
        [[PY, "-m", "churnguard.data"]],
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
    args = parser.parse_args()

    for cmd in COMMANDS[args.command][1]:
        print(f"$ {' '.join(cmd)}", flush=True)
        code = subprocess.call(cmd, env={**os.environ, "PYTHONUTF8": "1"})
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())

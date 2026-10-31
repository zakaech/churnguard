"""Chargement de la configuration centrale (config.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Lit le fichier YAML de configuration et le renvoie sous forme de dictionnaire."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str | Path) -> Path:
    """Résout un chemin de la configuration par rapport à la racine du projet."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p

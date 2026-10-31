"""Télécharge le jeu de données Telco Customer Churn depuis Kaggle vers data/raw/.

Authentification (une des options suivantes, voir .env.example) :
- KAGGLE_API_TOKEN (jeton généré sur https://www.kaggle.com/settings/api) ;
- KAGGLE_USERNAME + KAGGLE_KEY (ancienne clé API) ;
- fichier ~/.kaggle/access_token ou ~/.kaggle/kaggle.json.

Sans identifiants, le script explique comment placer le fichier à la main.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from churnguard.config import PROJECT_ROOT, load_config, resolve_path


def has_kaggle_credentials() -> bool:
    """Indique si des identifiants Kaggle sont disponibles (sans les afficher)."""
    if os.getenv("KAGGLE_API_TOKEN"):
        return True
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    kaggle_dir = Path(os.getenv("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    return (kaggle_dir / "access_token").exists() or (kaggle_dir / "kaggle.json").exists()


def manual_instructions(dataset: str, target: Path) -> str:
    return (
        "Aucun identifiant Kaggle trouvé.\n\n"
        "Option 1 — configurer l'API Kaggle :\n"
        "  1. Générer un jeton sur https://www.kaggle.com/settings/api\n"
        "  2. Copier .env.example en .env et renseigner KAGGLE_API_TOKEN\n"
        "     (ou KAGGLE_USERNAME + KAGGLE_KEY)\n"
        "  3. Relancer : python tasks.py download\n\n"
        "Option 2 — téléchargement manuel :\n"
        f"  1. Ouvrir https://www.kaggle.com/datasets/{dataset}\n"
        "  2. Cliquer sur « Download » et décompresser l'archive\n"
        f"  3. Placer le fichier CSV ici : {target}\n"
    )


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    cfg = load_config()
    target = resolve_path(cfg["paths"]["raw_data"])
    dataset = cfg["kaggle"]["dataset"]

    if target.exists():
        print(f"Fichier déjà présent, rien à faire : {target}")
        return 0

    if not has_kaggle_credentials():
        print(manual_instructions(dataset, target))
        return 1

    # Import tardif : le module kaggle tente de s'authentifier dès son import.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Téléchargement de {dataset} vers {target.parent} ...")
    api.dataset_download_files(dataset, path=str(target.parent), unzip=True, quiet=False)

    if not target.exists():
        print(f"Téléchargement terminé mais fichier attendu introuvable : {target}")
        return 1
    print(f"OK : {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

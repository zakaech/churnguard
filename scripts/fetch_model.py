"""Télécharge le modèle final depuis une release GitHub et vérifie son empreinte SHA-256.

Un fichier joblib est un « pickle » : le charger exécute du code. On ne l'installe donc que
si son empreinte SHA-256 correspond exactement à celle enregistrée dans config.yaml
(model.release.sha256). Sinon, le fichier est rejeté et rien n'est écrit.

Usage :
  python scripts/fetch_model.py                     # télécharge vers models/
  python scripts/fetch_model.py --sha256 models/churnguard_model.joblib   # calcule l'empreinte

Dépôt privé : définir GITHUB_TOKEN (jeton avec accès en lecture au dépôt).
Aucune dépendance hors bibliothèque standard (utilisable dans une image légère).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.github.com"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def install_verified(downloaded: Path, expected_sha256: str, destination: Path) -> None:
    """Déplace le fichier téléchargé vers `destination` seulement si l'empreinte correspond."""
    actual = sha256_of(downloaded)
    if actual != expected_sha256.lower():
        downloaded.unlink(missing_ok=True)
        raise ValueError(
            f"Empreinte SHA-256 incorrecte pour {destination.name} : attendu "
            f"{expected_sha256}, obtenu {actual}. Fichier rejeté."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(downloaded, destination)


def _request(url: str, token: str | None, accept: str) -> urllib.request.Request:
    headers = {"Accept": accept, "User-Agent": "churnguard-fetch-model"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def asset_urls(repo: str, tag: str, token: str | None) -> dict[str, str]:
    """URL de téléchargement (API GitHub) de chaque fichier de la release."""
    with urllib.request.urlopen(
        _request(f"{API}/repos/{repo}/releases/tags/{tag}", token, "application/vnd.github+json"),
        timeout=30,
    ) as r:
        release = json.load(r)
    return {asset["name"]: asset["url"] for asset in release["assets"]}


def download(url: str, token: str | None, target: Path) -> None:
    with urllib.request.urlopen(_request(url, token, "application/octet-stream"), timeout=120) as r:
        target.write_bytes(r.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sha256", type=Path, help="Affiche l'empreinte SHA-256 d'un fichier")
    parser.add_argument("--dest", type=Path, default=ROOT / "models")
    args = parser.parse_args()

    if args.sha256:
        print(sha256_of(args.sha256))
        return 0

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    release = cfg["model"]["release"]
    token = os.getenv("GITHUB_TOKEN")
    urls = asset_urls(release["repo"], release["tag"], token)
    for name in ("churnguard_model.joblib", "metadata.json"):
        if name not in urls:
            print(f"Fichier {name} absent de la release {release['tag']}.", file=sys.stderr)
            return 1

    with tempfile.TemporaryDirectory() as tmp:
        model_tmp = Path(tmp) / "model.joblib"
        download(urls["churnguard_model.joblib"], token, model_tmp)
        install_verified(model_tmp, release["sha256"], args.dest / "churnguard_model.joblib")
        download(urls["metadata.json"], token, args.dest / "metadata.json")
    print(f"Modèle vérifié (SHA-256 {release['sha256'][:12]}…) et installé dans {args.dest}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

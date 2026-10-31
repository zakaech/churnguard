"""Tests des scripts : vérification SHA-256 du modèle et modèle de test pour la CI."""

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from churnguard.api.main import create_app
from churnguard.api.schemas import EXAMPLE_CUSTOMER

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_install_verified_accepts_matching_checksum(tmp_path):
    fetch = load_script("fetch_model")
    downloaded, dest = tmp_path / "dl.bin", tmp_path / "models" / "model.joblib"
    downloaded.write_bytes(b"contenu du modele")
    fetch.install_verified(downloaded, hashlib.sha256(b"contenu du modele").hexdigest(), dest)
    assert dest.read_bytes() == b"contenu du modele"


def test_install_verified_rejects_wrong_checksum(tmp_path):
    fetch = load_script("fetch_model")
    downloaded, dest = tmp_path / "dl.bin", tmp_path / "models" / "model.joblib"
    downloaded.write_bytes(b"fichier modifie")
    with pytest.raises(ValueError, match="SHA-256"):
        fetch.install_verified(downloaded, "0" * 64, dest)
    assert not dest.exists()
    assert not downloaded.exists()


def test_make_test_model_is_served_by_api(tmp_path):
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_test_model.py"), "--out", str(tmp_path)],
        check=True,
        cwd=ROOT,
    )
    with TestClient(create_app(tmp_path / "churnguard_model.joblib", api_key="")) as client:
        assert client.get("/health").json()["model_version"] == "ci-test"
        r = client.post("/predict", json=EXAMPLE_CUSTOMER)
        assert r.status_code == 200
        assert len(r.json()["top_factors"]) == 5
        assert client.get("/model/info").status_code == 200

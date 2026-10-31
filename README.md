# ChurnGuard

Prédiction et explication du risque de départ (churn) des clients d'un opérateur télécom.

> Projet en cours de construction — README complet prévu en phase 5.

## Données

- Jeu de données : **Telco Customer Churn** (IBM), disponible sur Kaggle :
  `blastchar/telco-customer-churn`.
- Licence : **à compléter** (à vérifier sur la page Kaggle avant publication du dépôt).
- Les données ne sont pas versionnées. Les tests utilisent un échantillon **synthétique**
  (`tests/data/sample.csv`) généré par `scripts/make_synthetic_sample.py`.

## Installation

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate          # Linux/macOS : source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

## Commandes

```bash
python tasks.py download        # télécharge les données (API Kaggle, ou instructions manuelles)
python tasks.py data            # nettoie et découpe (train/val/test) vers data/processed/
python tasks.py notebook        # ré-exécute le notebook d'exploration
python tasks.py train-baseline  # entraîne les baselines, métriques sur la validation
python tasks.py test            # tests pytest
python tasks.py lint            # ruff
```

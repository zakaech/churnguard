---
title: ChurnGuard
sdk: docker
app_port: 7860
---

# ChurnGuard — démo (Hugging Face Space)

> Ce fichier est le **README du Space** (l'en-tête YAML ci-dessus configure le Space).
> Il n'est pas déployé tant que la licence des données n'est pas confirmée.

Interface de prédiction et d'explication du risque de départ client. L'API tourne dans le
même conteneur, sur `127.0.0.1:8000` : elle n'est pas accessible depuis Internet ; seule
l'interface (port 7860) l'est.

## Préparer le déploiement (à faire seulement après confirmation de la licence)

1. **Publier le modèle** : créer la release `model-v1.0.0` du dépôt GitHub avec
   `models/churnguard_model.joblib` et `models/metadata.json`. Vérifier que l'empreinte
   (`python scripts/fetch_model.py --sha256 models/churnguard_model.joblib`) est identique à
   `model.release.sha256` dans `config.yaml`.
2. **Créer le Space** (SDK Docker, matériel « CPU Basic »), puis y copier :
   - ce fichier comme `README.md` du Space ;
   - `docker/space.Dockerfile` renommé en `Dockerfile` ;
   - `docker/space-start.sh`, `config.yaml`, `requirements-api.txt`, `requirements-ui.txt`,
     `constraints.txt`, `src/churnguard/` et `scripts/fetch_model.py` (mêmes chemins).
3. **Secrets du Space** (Settings → Variables and secrets) :
   - `GITHUB_TOKEN` : jeton en lecture seule sur le dépôt (nécessaire si le dépôt est privé) ;
   - `CHURNGUARD_API_KEY` : facultatif (sinon une clé est générée au démarrage ; l'API étant
     interne, elle ne sort jamais du conteneur).
4. Au démarrage, `space-start.sh` télécharge le modèle, **vérifie son SHA-256** (refus sinon),
   lance l'API sur 127.0.0.1:8000, attend qu'elle soit prête, puis lance l'interface sur 7860.

## Points non vérifiés sur la documentation Hugging Face (31/10/2026)

- Mise en veille d'un Space gratuit inactif (délai, temps de réveil).
- Possibilité de rendre le Space privé sur l'offre gratuite.

Ressources mesurées en local : environ 180 Mo pour l'API et 60 Mo pour l'interface en pic,
bien en dessous du matériel gratuit annoncé (« CPU Basic : 2 vCPU, 16 GB »).

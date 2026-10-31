# Image combinée pour Hugging Face Spaces (SDK Docker) : API interne + interface publique.
# Un seul port exposé (7860) : seule l'interface est accessible, l'API reste sur 127.0.0.1.
# Les Spaces exécutent le conteneur avec l'UID 1000 (documentation Hugging Face).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=60 \
    PIP_RETRIES=10 \
    PYTHONPATH=/app/src \
    HOME=/home/user \
    PORT=7860

RUN useradd --create-home --uid 1000 user
WORKDIR /app

COPY requirements-api.txt requirements-ui.txt constraints.txt ./
RUN pip install -r requirements-api.txt -r requirements-ui.txt -c constraints.txt

COPY config.yaml ./
COPY src/churnguard ./src/churnguard
COPY scripts/fetch_model.py ./scripts/
COPY docker/space-start.sh ./docker/

# Dossier du modèle accessible en écriture à l'utilisateur (téléchargement au démarrage).
RUN mkdir -p /app/models && chown user:user /app/models
USER user

EXPOSE 7860
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD ["python", "-c", "import urllib.request, sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:7860/_stcore/health', timeout=4).status != 200)"]

CMD ["sh", "docker/space-start.sh"]

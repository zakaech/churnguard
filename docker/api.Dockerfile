# Image de l'API ChurnGuard : légère, utilisateur non-root, modèle monté au lancement.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    CHURNGUARD_MODEL_PATH=/app/models/churnguard_model.joblib

WORKDIR /app

COPY requirements-api.txt constraints.txt ./
RUN pip install -r requirements-api.txt -c constraints.txt

# Code : le package est importé via PYTHONPATH ; config.yaml est lu à la racine (/app).
COPY config.yaml ./
COPY src/churnguard ./src/churnguard

RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
    CMD ["python", "-c", "import urllib.request, sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status != 200)"]

CMD ["uvicorn", "churnguard.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

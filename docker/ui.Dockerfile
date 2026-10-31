# Image de l'interface Streamlit : elle appelle l'API (API_URL) et ne contient pas le modèle.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=60 \
    PIP_RETRIES=10 \
    PYTHONPATH=/app/src \
    API_URL=http://api:8000

WORKDIR /app

COPY requirements-ui.txt constraints.txt ./
RUN pip install -r requirements-ui.txt -c constraints.txt

# Seulement les modules utiles à l'interface (pas de modèle, pas de SHAP).
COPY src/churnguard/__init__.py src/churnguard/labels.py ./src/churnguard/
COPY src/churnguard/ui ./src/churnguard/ui

RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import urllib.request, sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status != 200)"]

CMD ["streamlit", "run", "src/churnguard/ui/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", \
     "--browser.gatherUsageStats=false"]

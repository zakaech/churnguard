#!/bin/sh
# Démarrage de l'image combinée (Hugging Face Spaces) : API interne + interface publique.
# - L'API écoute sur 127.0.0.1:8000 : elle n'est PAS joignable depuis l'extérieur.
# - L'interface Streamlit écoute sur le seul port exposé ($PORT, 7860 par défaut).
set -eu

MODEL="${CHURNGUARD_MODEL_PATH:-/app/models/churnguard_model.joblib}"
export CHURNGUARD_MODEL_PATH="$MODEL"

# Modèle absent : téléchargement depuis la release GitHub, avec vérification SHA-256
# (GITHUB_TOKEN à définir comme secret du Space si le dépôt est privé).
if [ ! -f "$MODEL" ]; then
    echo "Modèle absent : téléchargement depuis la release GitHub..."
    python scripts/fetch_model.py --dest "$(dirname "$MODEL")"
fi

# Clé d'API : fournie (secret du Space) ou générée au démarrage, partagée par les deux
# processus. L'API étant interne, la clé ne sort jamais du conteneur.
if [ -z "${CHURNGUARD_API_KEY:-}" ]; then
    CHURNGUARD_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
fi
export CHURNGUARD_API_KEY
export API_URL="http://127.0.0.1:8000"

uvicorn churnguard.api.main:app --host 127.0.0.1 --port 8000 &

python - <<'EOF'
import sys, time, urllib.request
for _ in range(90):
    try:
        if urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2).status == 200:
            print("API prête.")
            sys.exit(0)
    except OSError:
        time.sleep(1)
sys.exit("L'API n'a pas démarré dans le délai prévu.")
EOF

exec streamlit run src/churnguard/ui/app.py \
    --server.address=0.0.0.0 --server.port="${PORT:-7860}" \
    --server.headless=true --browser.gatherUsageStats=false

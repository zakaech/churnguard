# Interface MLflow optionnelle (profil « mlflow ») pour consulter les expériences locales.
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY constraints.txt /tmp/constraints.txt
RUN pip install mlflow==3.17.2 -c /tmp/constraints.txt

RUN useradd --create-home --uid 10001 app
USER app
WORKDIR /mlflow

EXPOSE 5000
CMD ["mlflow", "ui", "--backend-store-uri", "sqlite:////mlflow/mlflow.db", "--host", "0.0.0.0", "--port", "5000"]

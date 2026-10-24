# Own MLflow server image: native arm64 for Apple Silicon, and pinned to the
# same MLflow version as the training and serving code.
FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN pip install mlflow==2.17.2 sqlalchemy==2.0.36 \
 && useradd --uid 10001 --create-home mlflow \
 && mkdir -p /mlflow && chown 10001 /mlflow
USER 10001
EXPOSE 5000

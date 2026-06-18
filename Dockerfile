FROM python:3.11-slim

# libgomp1 est requis par LightGBM (dépendance OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Installation de uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copie des fichiers de dépendances seuls, pour profiter du cache Docker
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copie uniquement du code nécessaire à l'exécution
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 7860

ENV PYTHONUNBUFFERED=1

CMD ["uv", "run", "python", "src/app.py"]
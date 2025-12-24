# syntax=docker/dockerfile:1

FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    curl build-essential git && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 - --yes

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY pyproject.toml poetry.lock* /app/

RUN poetry install --only main --no-root

COPY src/backend /app/backend
COPY src/core /app/core

EXPOSE 8000

# Run FastAPI using uvicorn
CMD ["poetry", "run", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

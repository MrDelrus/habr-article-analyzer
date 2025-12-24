# syntax=docker/dockerfile:1

FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    curl build-essential git && \
    rm -rf /var/lib/apt/lists/*

ENV POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app

COPY pyproject.toml poetry.lock* /app/
COPY src/backend /app/backend
COPY src/core /app/core

RUN poetry install --only main

EXPOSE 8000

# Run FastAPI using uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

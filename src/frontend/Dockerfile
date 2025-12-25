# syntax=docker/dockerfile:1

FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    curl build-essential git && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 - --yes

ENV PATH="/root/.local/bin:$PATH"
ENV POETRY_VIRTUALENVS_CREATE=false
WORKDIR /app
ENV PYTHONPATH=/app

COPY src/frontend/pyproject.toml src/frontend/poetry.lock* /app/

RUN poetry install --only main --no-root

COPY src/frontend/frontend /app/frontend
COPY src/core /app/core

EXPOSE 8502

CMD ["poetry", "run", "streamlit", "run", "frontend/app.py", "--server.port=8502", "--server.address=0.0.0.0", "--server.headless=true"]

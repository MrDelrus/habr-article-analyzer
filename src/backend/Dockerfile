# syntax=docker/dockerfile:1

FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    curl build-essential git && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y locales && \
    locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

RUN curl -sSL https://install.python-poetry.org | python3 - --yes

ENV PATH="/root/.local/bin:$PATH"
ENV POETRY_VIRTUALENVS_CREATE=false
WORKDIR /app

COPY src/backend/pyproject.toml src/backend/poetry.lock* /app/

RUN poetry install --only main --no-root

COPY src/backend/backend /app/backend
COPY src/core /app/core

EXPOSE 8000

# Run FastAPI via uvicorn using Poetry environment
CMD ["poetry", "run", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

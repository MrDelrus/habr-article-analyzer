# syntax=docker/dockerfile:1

FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    curl build-essential git && \
    apt-get install -y locales && \
    rm -rf /var/lib/apt/lists/*

# setting locales to en_US.UTF-8 for models work with texts
RUN sed -i '/en_US.UTF-8/s/^# //' /etc/locale.gen && \
    locale-gen

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8


RUN curl -sSL https://install.python-poetry.org | python3 - --yes

ENV PATH="/root/.local/bin:$PATH"
ENV POETRY_VIRTUALENVS_CREATE=false
WORKDIR /app

COPY src/ml_service/pyproject.toml src/ml_service/poetry.lock* /app/

RUN poetry install --only main --no-root

COPY src/ml_service/ml_service /app/ml_service
COPY src/core /app/core
COPY models_cache models_cache

EXPOSE 8000

# Run FastAPI via uvicorn using Poetry environment
CMD ["poetry", "run", "uvicorn", "ml_service.main:app", "--host", "0.0.0.0", "--port", "8000"]

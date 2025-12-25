FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml poetry.lock* ./

RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-root --only main

COPY app.py .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8502", "--server.address=0.0.0.0"]

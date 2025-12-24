FROM python:3.11-slim

WORKDIR /app

# Копируем файлы зависимостей
COPY pyproject.toml poetry.lock* ./

# Устанавливаем Poetry и зависимости без установки проекта
RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-root --only main

# Копируем приложение
COPY app.py .

# Запускаем Streamlit
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8502", "--server.address=0.0.0.0"]

# Используем официальный образ Python в качестве базового
FROM python:3.12

# Устанавливаем переменные среды
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Создаем рабочую директорию
WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libatlas-base-dev \
    liblapack-dev \
    pkg-config \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Копируем файл зависимостей
COPY requirements.txt /app/

# Устанавливаем зависимости
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Устанавливаем модель spaCy
RUN python -m spacy download en_core_web_sm

# Копируем остальной код приложения
COPY . /app/

# Собираем статические файлы
RUN python manage.py collectstatic --noinput

# Выполняем миграции
RUN python manage.py migrate

# Открываем порт 8000
EXPOSE 8000

# Устанавливаем команду запуска приложения
CMD ["gunicorn", "grinatom.wsgi:application", "--bind", "0.0.0.0:8000"]

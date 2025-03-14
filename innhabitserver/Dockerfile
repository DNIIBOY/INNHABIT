FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /code
COPY requirements.txt /code/

RUN apt update && apt install -y \
    gcc \
    python3-dev \
    libpq-dev \
    gettext \
    postgresql \
    postgresql-contrib \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY . /code/

CMD ["gunicorn", "--bind", ":8000", "--workers", "2", "innhabitserver.wsgi:application"]

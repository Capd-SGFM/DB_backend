FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libpq-dev gcc postgresql-client && rm -rf /var/lib/apt/lists/*

COPY ./DB_backend/ .
RUN pip install --no-cache-dir -r requirements.txt



CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

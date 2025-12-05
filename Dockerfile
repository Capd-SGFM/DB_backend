FROM python:3.12-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y libpq-dev gcc postgresql-client && rm -rf /var/lib/apt/lists/*

COPY ./DB_backend/ .
RUN pip install --no-cache-dir -r requirements.txt

# Development stage with hot reload
FROM base AS dev
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base AS prod
# Optional: Install GPU support (cudf) - comment out if not needed
# RUN pip install --no-cache-dir --extra-index-url=https://pypi.nvidia.com cudf-cu12
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# FinSight FastAPI Production Container (Task J-6)
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for building wheels if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install unified Python dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application modules and processed corpus/index
COPY api/ ./api/
COPY retrieval/ ./retrieval/
COPY generation/ ./generation/
COPY data/processed/ ./data/processed/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

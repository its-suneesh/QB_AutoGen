FROM python:3.11-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

RUN mkdir -p logs/app logs/access logs/error logs/security && \
    chown -R appuser:appuser /app

USER appuser

# Documentation only - the real port is PORT in .env, read by run.py.
# Publishing it is docker-compose's job (or -p on docker run).
EXPOSE 9000

ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
# A container has to listen on every interface; .env's loopback default is
# for bare-metal runs. docker-compose sets this too.
ENV HOST=0.0.0.0

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f "http://localhost:${PORT:-9000}/health" || exit 1

    
# run.py binds HOST/PORT/WORKERS from .env, so this line never needs editing.
CMD ["python", "run.py"]
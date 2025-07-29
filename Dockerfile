# Dockerfile

# --- Build Stage ---
FROM python:3.11-slim as builder


WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    [cite_start]pip install --no-cache-dir -r requirements.txt [cite: 9]



FROM python:3.11-slim


WORKDIR /app


RUN useradd --create-home appuser
USER appuser


COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin


COPY . .

EXPOSE 5000


ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production


CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:5000", "run:app"]

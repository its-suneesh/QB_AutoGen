# Dockerfile

# --- Build Stage ---
# Use a slim Python image as a base for a smaller final image size.
FROM python:3.11-slim as builder

# Set the working directory inside the container.
WORKDIR /app

# Install build dependencies that might be needed by some Python packages.
RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies.
# This is done in a separate layer to leverage Docker's caching.
# The layer will only be rebuilt if requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# --- Final Stage ---
# Use the same slim Python base image.
FROM python:3.11-slim

# Set the working directory.
WORKDIR /app

# Create a non-root user for security purposes.
RUN useradd --create-home appuser
USER appuser

# Copy the installed packages from the builder stage.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the application code into the container.
COPY . .

# Expose the port the app runs on.
EXPOSE 5000

# Set environment variables for production.
# PYTHONUNBUFFERED ensures that Python output is sent straight to the terminal
# (and Docker logs) without being buffered first.
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# The command to run the application using Gunicorn, a production-grade WSGI server.
# Binds the server to all network interfaces on port 5000.
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:5000", "run:app"]
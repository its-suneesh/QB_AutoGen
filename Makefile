# Makefile for QB AutoGen Docker operations

.PHONY: help build up down logs clean dev restart health

# Default target
help:
	@echo "Available commands:"
	@echo "  build     - Build the Docker image"
	@echo "  up        - Start the application with Gunicorn"
	@echo "  dev       - Start the application in development mode"
	@echo "  down      - Stop the application"
	@echo "  logs      - Show application logs"
	@echo "  restart   - Restart the application"
	@echo "  health    - Check application health"
	@echo "  clean     - Remove containers, images, and volumes"
	@echo "  status    - Show container status"

# Build the Docker image
build:
	docker-compose build

# Start with Gunicorn (production)
up:
	docker-compose up -d

# Start in development mode with hot reload
dev:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Stop the application
down:
	docker-compose down

# Show logs
logs:
	docker-compose logs -f qb-autogen-api

# Restart the application
restart:
	docker-compose restart qb-autogen-api

# Check application health
health:
	@echo "Checking application health..."
	@curl -f http://localhost:5000/health || echo "Health check failed"

# Clean up everything
clean:
	docker-compose down -v --rmi all --remove-orphans
	docker system prune -f

# Install dependencies locally (for development)
install:
	pip install -r requirements.txt

# Run tests (if you add them later)
test:
	docker-compose exec qb-autogen-api python -m pytest

# View container status
status:
	docker-compose ps

# Follow logs in real-time
tail:
	docker-compose logs -f --tail=100 qb-autogen-api
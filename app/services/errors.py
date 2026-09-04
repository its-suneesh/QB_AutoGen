"""The one exception the API layer turns into an HTTP status."""

from __future__ import annotations


class ServiceError(Exception):
    """Custom exception for service layer errors."""
    def __init__(self, message, status_code=503):
        super().__init__(message)
        self.status_code = status_code

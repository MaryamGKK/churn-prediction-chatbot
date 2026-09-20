from __future__ import annotations

import time
import uuid
from collections import defaultdict

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.config import API_KEY

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    if not API_KEY:
        return
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class RateLimiter:
    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self._timestamps: dict[str, list[float]] = defaultdict(list)
        self._max = max_requests
        self._window = window_seconds

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        recent = [t for t in self._timestamps[key] if now - t < self._window]
        self._timestamps[key] = recent
        if len(recent) >= self._max:
            return False
        recent.append(now)
        return True


class SecurityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rate_limiter: RateLimiter | None = None):
        super().__init__(app)
        self.rate_limiter = rate_limiter or RateLimiter()

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        client_ip = request.client.host if request.client else "unknown"
        if not self.rate_limiter.allow(client_ip):
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests"},
                headers={"Retry-After": "60", "X-Request-ID": request_id},
            )

        response = await call_next(request)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Cache-Control"] = "no-store"
        return response

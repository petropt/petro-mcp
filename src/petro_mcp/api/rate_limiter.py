"""RapidAPI proxy-secret validation and per-user rate limiting middleware.

RapidAPI forwards every request with:
  - ``X-RapidAPI-Proxy-Secret`` – shared secret configured on the provider side
  - ``X-RapidAPI-User``         – the subscriber's RapidAPI username

This module provides a Starlette middleware that:
1. Validates the proxy secret (rejects requests with a mismatched value).
2. Tracks per-user request counts using a simple in-memory sliding window.
3. Injects ``X-RateLimit-Limit``, ``X-RateLimit-Remaining``, and
   ``X-RateLimit-Reset`` response headers.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Proxy secret – set via environment variable in production.
RAPIDAPI_PROXY_SECRET: str = os.getenv("RAPIDAPI_PROXY_SECRET", "")

# Default daily limit applied when no tier-specific header is present.
DEFAULT_DAILY_LIMIT: int = int(os.getenv("RAPIDAPI_DEFAULT_DAILY_LIMIT", "100"))

# Tier limits (requests per day).  RapidAPI itself enforces limits, but we
# mirror them so we can return accurate headers and protect the backend if
# RapidAPI's enforcement lags.
TIER_LIMITS: dict[str, int] = {
    "free": 100,
    "basic": 10_000,
    "pro": 100_000,
    "ultra": 1_000_000_000,  # effectively unlimited
}

# Paths that bypass authentication (health checks, docs).
_PUBLIC_PATHS: set[str] = {"/health", "/openapi.json", "/docs", "/redoc", "/api/v1/docs"}


# ---------------------------------------------------------------------------
# In-memory usage tracker (replace with Redis for multi-instance deployments)
# ---------------------------------------------------------------------------

class _UsageTracker:
    """Thread-safe-ish in-memory daily request counter.

    Keyed on RapidAPI username.  Resets counters at the start of each UTC day.
    """

    def __init__(self) -> None:
        self._counts: dict[str, int] = defaultdict(int)
        self._day: int = self._current_day()

    @staticmethod
    def _current_day() -> int:
        return int(time.time() // 86400)

    def _maybe_reset(self) -> None:
        today = self._current_day()
        if today != self._day:
            self._counts.clear()
            self._day = today

    def increment(self, user: str) -> int:
        """Increment and return the new count for *user*."""
        self._maybe_reset()
        self._counts[user] += 1
        return self._counts[user]

    def count(self, user: str) -> int:
        self._maybe_reset()
        return self._counts[user]

    def reset(self) -> None:
        """Force-clear all counters (useful in tests)."""
        self._counts.clear()
        self._day = self._current_day()


# Singleton instance used by the middleware.
usage_tracker = _UsageTracker()


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class RapidAPIMiddleware(BaseHTTPMiddleware):
    """Starlette middleware for RapidAPI proxy-secret validation and rate
    limiting.

    When ``RAPIDAPI_PROXY_SECRET`` is empty the middleware is effectively a
    pass-through (useful during local development).
    """

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        # Let public paths through unconditionally.
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        # --- Proxy-secret validation ---
        proxy_secret = os.getenv("RAPIDAPI_PROXY_SECRET", "")
        if proxy_secret:
            incoming_secret = request.headers.get("x-rapidapi-proxy-secret", "")
            if incoming_secret != proxy_secret:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Invalid RapidAPI proxy secret."},
                )

        # --- Rate limiting ---
        user = request.headers.get("x-rapidapi-user", "anonymous")
        subscription = request.headers.get("x-rapidapi-subscription", "free").lower()
        limit = TIER_LIMITS.get(subscription, DEFAULT_DAILY_LIMIT)

        current_count = usage_tracker.increment(user)
        remaining = max(0, limit - current_count)

        if current_count > limit:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        "Rate limit exceeded. Upgrade your plan at "
                        "https://rapidapi.com/petropt/api/petro-mcp"
                    ),
                    "upgrade_url": "https://rapidapi.com/petropt/api/petro-mcp",
                },
                headers={
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(self._seconds_until_midnight()),
                },
            )

        # --- Proceed with the request ---
        response: Response = await call_next(request)

        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(self._seconds_until_midnight())

        return response

    @staticmethod
    def _seconds_until_midnight() -> int:
        now = time.time()
        next_day = (int(now // 86400) + 1) * 86400
        return int(next_day - now)

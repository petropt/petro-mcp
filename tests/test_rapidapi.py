"""Tests for RapidAPI rate limiter middleware and /api/v1/docs catalog endpoint."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from petro_mcp.api.app import app
from petro_mcp.api.rate_limiter import usage_tracker, TIER_LIMITS

client = TestClient(app)


# ---------------------------------------------------------------------------
# Rate limiter unit tests
# ---------------------------------------------------------------------------

class TestUsageTracker:
    """Unit tests for the in-memory usage tracker."""

    def test_increment_and_count(self):
        from petro_mcp.api.rate_limiter import _UsageTracker
        tracker = _UsageTracker()
        assert tracker.count("alice") == 0
        assert tracker.increment("alice") == 1
        assert tracker.increment("alice") == 2
        assert tracker.count("alice") == 2

    def test_separate_users(self):
        from petro_mcp.api.rate_limiter import _UsageTracker
        tracker = _UsageTracker()
        tracker.increment("alice")
        tracker.increment("bob")
        tracker.increment("bob")
        assert tracker.count("alice") == 1
        assert tracker.count("bob") == 2

    def test_reset_clears_counts(self):
        from petro_mcp.api.rate_limiter import _UsageTracker
        tracker = _UsageTracker()
        tracker.increment("alice")
        tracker.increment("alice")
        tracker.reset()
        assert tracker.count("alice") == 0

    def test_day_rollover_resets(self):
        from petro_mcp.api.rate_limiter import _UsageTracker
        tracker = _UsageTracker()
        tracker.increment("alice")
        assert tracker.count("alice") == 1
        # Simulate a day change by manipulating the internal day marker.
        tracker._day -= 1
        assert tracker.count("alice") == 0


# ---------------------------------------------------------------------------
# Middleware integration tests
# ---------------------------------------------------------------------------

class TestRapidAPIMiddleware:
    """Integration tests for RapidAPI proxy-secret validation and rate headers."""

    def setup_method(self):
        usage_tracker.reset()

    def test_health_bypasses_middleware(self):
        """Public paths like /health should work without any RapidAPI headers."""
        r = client.get("/health")
        assert r.status_code == 200

    def test_rate_limit_headers_present(self):
        """Successful API calls should include rate-limit response headers."""
        r = client.post("/api/v1/drilling/hydrostatic", json={
            "mud_weight_ppg": 10.0, "tvd_ft": 10000.0,
        })
        assert r.status_code == 200
        assert "x-ratelimit-limit" in r.headers
        assert "x-ratelimit-remaining" in r.headers
        assert "x-ratelimit-reset" in r.headers

    @patch.dict(os.environ, {"RAPIDAPI_PROXY_SECRET": "test-secret-abc"})
    def test_proxy_secret_rejected_when_missing(self):
        """If RAPIDAPI_PROXY_SECRET is set, requests without it are rejected."""
        r = client.post("/api/v1/drilling/hydrostatic", json={
            "mud_weight_ppg": 10.0, "tvd_ft": 10000.0,
        })
        assert r.status_code == 403

    @patch.dict(os.environ, {"RAPIDAPI_PROXY_SECRET": "test-secret-abc"})
    def test_proxy_secret_accepted_when_correct(self):
        """Requests with the correct proxy secret should pass."""
        r = client.post(
            "/api/v1/drilling/hydrostatic",
            json={"mud_weight_ppg": 10.0, "tvd_ft": 10000.0},
            headers={"X-RapidAPI-Proxy-Secret": "test-secret-abc"},
        )
        assert r.status_code == 200

    def test_rate_limit_exceeded(self):
        """Exceeding the daily limit should return 429."""
        # Pre-fill the counter to exhaust the free tier.
        free_limit = TIER_LIMITS["free"]
        user = "test-ratelimit-user"
        for _ in range(free_limit):
            usage_tracker.increment(user)

        # The next request should be rejected.
        r = client.post(
            "/api/v1/drilling/hydrostatic",
            json={"mud_weight_ppg": 10.0, "tvd_ft": 10000.0},
            headers={"X-RapidAPI-User": user},
        )
        assert r.status_code == 429
        body = r.json()
        assert "Rate limit exceeded" in body["detail"]
        assert body["upgrade_url"] == "https://rapidapi.com/petropt/api/petro-mcp"


# ---------------------------------------------------------------------------
# /api/v1/docs catalog endpoint
# ---------------------------------------------------------------------------

class TestAPICatalog:
    """Tests for the /api/v1/docs endpoint catalog."""

    def setup_method(self):
        usage_tracker.reset()

    def test_catalog_returns_200(self):
        r = client.get("/api/v1/docs")
        assert r.status_code == 200

    def test_catalog_structure(self):
        r = client.get("/api/v1/docs")
        body = r.json()
        assert "total_endpoints" in body
        assert "endpoints" in body
        assert isinstance(body["endpoints"], list)
        assert body["total_endpoints"] == len(body["endpoints"])

    def test_catalog_contains_known_endpoints(self):
        r = client.get("/api/v1/docs")
        paths = [ep["path"] for ep in r.json()["endpoints"]]
        assert "/api/v1/decline/fit" in paths
        assert "/api/v1/pvt/properties" in paths
        assert "/api/v1/petrophys/archie" in paths
        assert "/api/v1/drilling/hydrostatic" in paths
        assert "/api/v1/economics/npv" in paths
        assert "/health" in paths

    def test_catalog_endpoint_has_tags(self):
        r = client.get("/api/v1/docs")
        for ep in r.json()["endpoints"]:
            if ep["path"] == "/api/v1/decline/fit":
                assert "DCA" in ep["tags"]
                break
        else:
            pytest.fail("/api/v1/decline/fit not found in catalog")

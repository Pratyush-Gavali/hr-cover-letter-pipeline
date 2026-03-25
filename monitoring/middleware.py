"""
monitoring/middleware.py

Starlette middleware that records HTTP metrics for every request
automatically. Add once to main.py — no per-route changes needed
for the HTTP layer.

Path normalisation prevents label cardinality explosion:
  /api/v1/covers/jd_1138  →  /api/v1/covers/{job_id}
  /api/v1/jobs/jd_9999    →  /api/v1/jobs/{job_id}
"""

from __future__ import annotations
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .metrics import HTTP_REQUESTS_TOTAL, HTTP_REQUEST_DURATION_SECONDS

_PATH_TEMPLATES = [
    ("/api/v1/covers/",     "/api/v1/covers/{job_id}"),
    ("/api/v1/jobs/",       "/api/v1/jobs/{job_id}"),
    ("/api/v1/applicants/", "/api/v1/applicants/{job_id}"),
]


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        endpoint = self._normalise(request.url.path)
        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=str(response.status_code),
        ).inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(
            method=request.method,
            endpoint=endpoint,
        ).observe(duration)

        return response

    @staticmethod
    def _normalise(path: str) -> str:
        for prefix, template in _PATH_TEMPLATES:
            if path.startswith(prefix) and len(path) > len(prefix):
                return template
        return path

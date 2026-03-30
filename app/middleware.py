from __future__ import annotations

import logging
import time
import uuid

from fastapi import Request, Response

from .metrics import HTTP_REQUEST_DURATION_SECONDS, HTTP_REQUESTS_TOTAL

logger = logging.getLogger("rag-api")


async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    start = time.perf_counter()

    try:
        response: Response = await call_next(request)
    except Exception:
        duration = time.perf_counter() - start
        path = request.url.path
        HTTP_REQUESTS_TOTAL.labels(request.method, path, "500").inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(request.method, path).observe(duration)
        logger.exception(
            "request_failed",
            extra={"request_id": request_id, "method": request.method, "path": path},
        )
        raise

    duration = time.perf_counter() - start
    path = request.url.path
    HTTP_REQUESTS_TOTAL.labels(request.method, path, str(response.status_code)).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(request.method, path).observe(duration)

    response.headers["x-request-id"] = request_id
    logger.info(
        "request",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": path,
            "status": response.status_code,
            "duration_ms": int(duration * 1000),
        },
    )
    return response

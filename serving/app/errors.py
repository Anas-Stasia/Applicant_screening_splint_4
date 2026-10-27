"""Error handling.

Every failure — bad payload, unscoreable text, oversized batch, model not loaded,
or an unexpected crash — comes back in one envelope:

    {"error": {"code": "TEXT_TOO_SHORT", "message": "...", "details": ..., "request_id": "..."}}

Client squads can branch on ``code`` (stable, machine-readable) and show
``message`` (human-readable), and quote ``request_id`` when reporting a bug so it
can be found in the server logs.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("screening.errors")


class AppError(Exception):
    status_code = 400
    code = "BAD_REQUEST"

    def __init__(self, message: str, *, details: dict | list | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class InputRejected(AppError):
    """Text passed schema validation but can't be scored (empty, too short, no words, too long)."""

    status_code = 422

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class UnknownCategory(AppError):
    status_code = 422
    code = "UNKNOWN_CATEGORY"


class BatchTooLarge(AppError):
    status_code = 413
    code = "BATCH_TOO_LARGE"


class ModelNotReady(AppError):
    status_code = 503
    code = "MODEL_NOT_READY"


class Unauthorized(AppError):
    status_code = 401
    code = "UNAUTHORIZED"


class InferenceFailed(AppError):
    status_code = 500
    code = "INFERENCE_FAILED"


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _envelope(request: Request, status: int, code: str, message: str, details=None) -> JSONResponse:
    body = {
        "error": {
            "code": code,
            "message": message,
            "details": details,
            "request_id": _request_id(request),
        }
    }
    return JSONResponse(status_code=status, content=jsonable_encoder(body))


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError):
        if exc.status_code >= 500:
            logger.error("request_id=%s %s: %s", _request_id(request), exc.code, exc.message)
        return _envelope(request, exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        # Trim pydantic's verbose error dicts to what a client can act on.
        details = [
            {"field": ".".join(str(p) for p in err.get("loc", ()) if p != "body"),
             "message": err.get("msg"),
             "type": err.get("type")}
            for err in exc.errors()
        ]
        return _envelope(request, 422, "VALIDATION_ERROR", "Request payload failed validation.", details)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(exc.status_code, "HTTP_ERROR")
        return _envelope(request, exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        # Log the full traceback server-side; never leak internals to the client.
        logger.exception("request_id=%s unhandled error", _request_id(request))
        return _envelope(request, 500, "INTERNAL_ERROR", "Unexpected server error.")

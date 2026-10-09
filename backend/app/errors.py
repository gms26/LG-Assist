"""Structured error format.

Every error response uses the same envelope so the frontend never has to
guess::

    {
      "error": {
        "code": "validation_error",
        "message": "human-readable summary",
        "details": [...],          # optional, machine-readable
        "request_id": "uuid"       # always present, for log correlation
      }
    }
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class ErrorCode:
    VALIDATION = "validation_error"
    NOT_FOUND = "not_found"
    BAD_REQUEST = "bad_request"
    UNAUTHORIZED = "unauthorized"
    DEPENDENCY = "dependency_unavailable"  # Ollama / DB / cloud LLM down
    TIMEOUT = "upstream_timeout"
    CONFLICT = "conflict"
    INTERNAL = "internal_error"


def error_response(
    status: int,
    code: str,
    message: str,
    details: list | dict | None = None,
    request_id: str | None = None,
) -> JSONResponse:
    body: dict = {"error": {"code": code, "message": message, "request_id": request_id}}
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(status_code=status, content=body)


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"field": ".".join(str(p) for p in e["loc"]), "message": e["msg"], "type": e["type"]}
            for e in exc.errors()
        ]
        return error_response(
            422, ErrorCode.VALIDATION, "Request validation failed.",
            details=details, request_id=_request_id(request),
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = ErrorCode.NOT_FOUND if exc.status_code == 404 else ErrorCode.BAD_REQUEST
        if exc.status_code == 401:
            code = ErrorCode.UNAUTHORIZED
        return error_response(
            exc.status_code, code, str(exc.detail) or "Request failed.",
            request_id=_request_id(request),
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        # Never leak tracebacks to clients; request_id lets us find it in logs.
        return error_response(
            500, ErrorCode.INTERNAL, "Something went wrong on our side.",
            request_id=_request_id(request),
        )

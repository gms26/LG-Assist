"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .db import init_db
from .errors import register_exception_handlers
from .middleware import RequestIDMiddleware
from .routes import health

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Best-effort: the app must still boot (and report degraded health)
    # when Postgres is unreachable, so init failures are swallowed here
    # and surfaced via /api/v1/health instead of crashing the process.
    try:
        init_db()
    except Exception as exc:  # noqa: BLE001 - deliberate: boot must not fail
        print(f"WARNING: database init failed ({type(exc).__name__}: {exc}); "
              "continuing in degraded mode, see /api/v1/health")
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)

# NOTE: endpoint paths/shapes are provisional until the owner's contract lands.
app.include_router(health.router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
def _root():
    return {"service": settings.app_name, "version": settings.app_version,
            "health": "/api/v1/health", "docs": "/api/docs"}

"""
main.py
=======
FastAPI app entrypoint.

Run with the RL repo's own venv interpreter (not system Python) so the
in-process env probe's `env`/`sim` imports resolve -- see config.py for why.
Run from the RL repo root (this file's grandparent directory):

    venv\\Scripts\\python.exe -m uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .routers import baseline as baseline_router
from .routers import episodes as episodes_router
from .routers import tests as tests_router
from .routers import training as training_router
from .routers import flighttest as flighttest_router

app = FastAPI(title="RL Glider Showcase Dashboard")

app.add_middleware(
    CORSMiddleware,
    # Any local dev-server port: Vite moves to 5174+ when 5173 is taken, and
    # a fixed origin then silently blocks every request ("Failed to fetch").
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tests_router.router)
app.include_router(episodes_router.router)
app.include_router(baseline_router.router)
app.include_router(training_router.router)
app.include_router(flighttest_router.router)


@app.get("/api/health")
def health() -> dict:
    return {
        "repo_ok": config.repo_ok(),
        "python_ok": config.python_ok(),
        "repo_root": str(config.GLIDER_REPO_ROOT),
    }

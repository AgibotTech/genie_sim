from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from .config import DreamConfig
from .contracts import (
    EvaluationRequest,
    ModelSelectionRequest,
    RunRequest,
    SessionRequest,
    StepRequest,
)
from .platform import Busy, Conflict, DreamPlatform


def create_app(config: DreamConfig, platform: DreamPlatform | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        app.state.platform = platform or DreamPlatform(config)
        try:
            yield
        finally:
            app.state.platform.close()

    app = FastAPI(title="Genie Sim Dream", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def protect(request: Request, call_next):
        token = os.environ.get(config.api_token_env)
        if token and request.url.path not in {"/", "/health/live", "/health/ready"}:
            supplied = request.headers.get("authorization", "")
            if not secrets.compare_digest(supplied, f"Bearer {token}"):
                return JSONResponse({"error": "Unauthorized"}, status_code=401)
        if request.method in {"POST", "PUT", "PATCH"}:
            content_length = request.headers.get("content-length")
            if content_length and (
                not content_length.isdigit() or int(content_length) > config.max_request_bytes
            ):
                return JSONResponse({"error": "Request too large"}, status_code=413)
            body = bytearray()
            async for block in request.stream():
                body.extend(block)
                if len(body) > config.max_request_bytes:
                    return JSONResponse({"error": "Request too large"}, status_code=413)
            request._body = bytes(body)
        return await call_next(request)

    @app.exception_handler(KeyError)
    async def missing(request, error):
        return JSONResponse({"error": "Resource not found"}, status_code=404)

    @app.exception_handler(Conflict)
    async def conflict(request, error):
        return JSONResponse({"error": str(error)}, status_code=409)

    @app.exception_handler(Busy)
    async def busy(request, error):
        return JSONResponse({"error": str(error)}, status_code=503)

    @app.exception_handler(ValueError)
    async def invalid(request, error):
        return JSONResponse({"error": str(error)}, status_code=400)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(Path(__file__).parent / "web/index.html")

    @app.get("/health/live")
    def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    def ready(request: Request):
        runtime = request.app.state.platform
        return JSONResponse(
            {"ready": runtime.ready, "error": runtime.startup_error},
            status_code=200 if runtime.ready else 503,
        )

    @app.get("/capabilities")
    def capabilities(request: Request):
        return request.app.state.platform.capabilities()

    @app.get("/models")
    def models(request: Request):
        return request.app.state.platform.model_status()

    @app.post("/models/active", status_code=202)
    def select_model(body: ModelSelectionRequest, request: Request):
        return request.app.state.platform.select_model(body.model_id)

    @app.get("/examples")
    def examples(request: Request):
        runtime = request.app.state.platform
        return {
            "examples": runtime.available_examples(),
            "rejected_count": len(runtime.rejected),
        }

    @app.post("/sessions", status_code=201)
    def create_session(body: SessionRequest, request: Request):
        return request.app.state.platform.create_session(body)

    @app.get("/sessions")
    def sessions(request: Request):
        return request.app.state.platform.list_sessions()

    @app.get("/sessions/{session_id}")
    def session(session_id: str, request: Request):
        return request.app.state.platform.session_info(session_id)

    @app.post("/sessions/{session_id}/steps", status_code=202)
    def step(session_id: str, body: StepRequest, request: Request):
        return request.app.state.platform.submit(session_id, body)

    @app.post("/sessions/{session_id}/run", status_code=202)
    def run(session_id: str, body: RunRequest, request: Request):
        return request.app.state.platform.submit(session_id, body, "run")

    @app.post("/sessions/{session_id}/evaluate", status_code=202)
    def evaluate(session_id: str, body: EvaluationRequest, request: Request):
        return request.app.state.platform.submit(session_id, body, "evaluate")

    @app.post("/sessions/{session_id}/stop")
    def stop(session_id: str, request: Request):
        return request.app.state.platform.stop_session(session_id)

    @app.post("/sessions/{session_id}/archive")
    def archive(session_id: str, request: Request):
        return request.app.state.platform.archive_session(session_id)

    @app.post("/sessions/{session_id}/finish")
    def finish(session_id: str, request: Request):
        return request.app.state.platform.finish(session_id)

    @app.delete("/sessions/{session_id}")
    def delete(session_id: str, request: Request):
        return request.app.state.platform.delete_session(session_id)

    @app.get("/jobs/{job_id}")
    def job(job_id: str, request: Request):
        runtime = request.app.state.platform
        with runtime.lock:
            return dict(runtime.jobs[job_id])

    @app.get("/artifacts/{session_id}/{relative_path:path}")
    def artifact(session_id: str, relative_path: str, request: Request):
        path = request.app.state.platform.artifact(session_id, relative_path)
        return FileResponse(path)

    return app

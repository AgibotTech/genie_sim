"""Resident HTTP evaluator for Dream finished-video task completion."""

from __future__ import annotations

import argparse
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse

from geniesim_dream_evaluation.dream_adapter import DreamVideoEvaluator
from geniesim_dream_evaluation.dream_contracts import EvaluationError
from geniesim_dream_evaluation.vlm_registry import DEFAULT_MODEL_KEY, default_models_config_path, load_model_registry

STATE: dict = {
    "ready": False,
    "error": None,
    "evaluator": None,
    "lock": threading.Lock(),
    "token_env": "DREAM_EVALUATOR_TOKEN",
    "models": {},
    "configured_models": {},
    "active_model_key": None,
}


def _check_auth(authorization: str | None) -> None:
    expected = os.environ.get(STATE["token_env"])
    if not expected:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise EvaluationError("unauthorized", "Bearer token required", http_status=401)
    token = authorization.removeprefix("Bearer ").strip()
    if token != expected:
        raise EvaluationError("unauthorized", "Invalid bearer token", http_status=401)


def create_app(
    *,
    shared_output_root: Path,
    models_config: Path | None = None,
    model_path: Path | None = None,
    token_env: str = "DREAM_EVALUATOR_TOKEN",
    load_on_startup: bool = True,
    load_model_fn=None,
) -> FastAPI:
    STATE["token_env"] = token_env
    STATE["models"] = {}
    STATE["configured_models"] = {}
    STATE["active_model_key"] = None
    config_path = (models_config or default_models_config_path()).resolve()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if load_on_startup:
            try:
                registry = load_model_registry(
                    config_path,
                    load_model_fn=load_model_fn,
                    force_model_path=model_path.resolve() if model_path is not None else None,
                    preload_keys=[DEFAULT_MODEL_KEY],
                    default_model_key=DEFAULT_MODEL_KEY,
                )
                STATE["evaluator"] = DreamVideoEvaluator(
                    shared_output_root=shared_output_root,
                    registry=registry,
                )
                STATE["models"] = registry.loaded_paths()
                STATE["configured_models"] = registry.configured_paths()
                STATE["active_model_key"] = registry.active_key
                STATE["ready"] = True
                STATE["error"] = None
            except Exception as error:  # noqa: BLE001
                STATE["ready"] = False
                STATE["error"] = f"{type(error).__name__}: {error}"
                raise
        yield

    app = FastAPI(title="Genie Sim Dream evaluator", lifespan=lifespan)

    @app.get("/health/live")
    def live():
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready():
        if STATE["ready"] and STATE["evaluator"] is not None:
            evaluator = STATE["evaluator"]
            registry = getattr(evaluator, "registry", None)
            models = registry.loaded_paths() if registry is not None else STATE["models"]
            if hasattr(evaluator, "enable_task_route") and hasattr(evaluator, "generic_prompt_mode"):
                judge_payload = {
                    "enable_task_route": evaluator.enable_task_route,
                    "generic_prompt_mode": evaluator.generic_prompt_mode,
                }
            elif registry is not None and hasattr(registry, "judge_flags"):
                judge_payload = registry.judge_flags
            else:
                judge_payload = None
            return {
                "ready": True,
                "models": models,
                "configured_models": STATE["configured_models"],
                "active_model_key": getattr(registry, "active_key", None) or STATE["active_model_key"],
                "judge": judge_payload,
            }
        return JSONResponse(
            status_code=503,
            content={"ready": False, "error": STATE["error"] or "model not loaded"},
        )

    @app.post("/evaluate")
    def evaluate(body: dict, authorization: str | None = Header(default=None)):
        try:
            _check_auth(authorization)
            if not STATE["ready"] or STATE["evaluator"] is None:
                raise EvaluationError("not_ready", "evaluator model is not ready", http_status=503)
            if not STATE["lock"].acquire(blocking=False):
                raise EvaluationError("evaluator_busy", "another evaluation is running", http_status=503)
            started = time.monotonic()
            try:
                result = STATE["evaluator"].evaluate(body)
                elapsed = round(time.monotonic() - started, 3)
                if isinstance(result, dict):
                    details = result.setdefault("details", {})
                    if isinstance(details, dict):
                        details["elapsed_seconds"] = elapsed
                        print(
                            f"[evaluate] elapsed_s={elapsed} "
                            f"success={details.get('success')} "
                            f"verdict={details.get('verdict')} "
                            f"evaluation_id={details.get('evaluation_id')}",
                            flush=True,
                        )
                    else:
                        print(f"[evaluate] elapsed_s={elapsed}", flush=True)
                else:
                    print(f"[evaluate] elapsed_s={elapsed}", flush=True)
                return result
            except EvaluationError as error:
                elapsed = round(time.monotonic() - started, 3)
                print(
                    f"[evaluate] elapsed_s={elapsed} error={error.code} "
                    f"evaluation_id={error.evaluation_id}",
                    flush=True,
                )
                raise
            finally:
                STATE["lock"].release()
        except EvaluationError as error:
            return JSONResponse(status_code=error.http_status, content=error.as_dict())

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve Genie Sim Dream video evaluator")
    parser.add_argument(
        "--models-config",
        type=Path,
        required=True,
        help="JSON mapping task families to separately downloaded VLM checkpoints",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional debug override: force every family to this single checkpoint",
    )
    parser.add_argument("--shared-output-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8981)
    parser.add_argument("--token-env", default="DREAM_EVALUATOR_TOKEN")
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    app = create_app(
        shared_output_root=args.shared_output_root.resolve(),
        models_config=args.models_config.resolve(),
        model_path=args.model_path.resolve() if args.model_path is not None else None,
        token_env=args.token_env,
    )
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()

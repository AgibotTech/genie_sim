# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Linux route manager: probe existing inference, connect tunnel workers, inspect, stop."""
import argparse
import asyncio
import concurrent.futures
import importlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone

import msgpack
from websockets.sync.client import connect

from common import (active_job, api, atomic_json, checked_url, get_job, locked,
                    owns_process, process_identity, read_state)

SCRIPT = Path(__file__).resolve()
from agent_identity import load_or_create_agent_id
from tunnel_client import TunnelClient, TunnelRejected

LOG = logging.getLogger("inference_route")


def now():
    return datetime.now(timezone.utc).isoformat()


def resolve_path(value, base):
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else base / path).resolve())


def load_config(path):
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text())
    models = config.get("models", [])
    if not models:
        raise ValueError("models must contain at least one existing inference service")
    names = set()
    for target in models:
        name = target["name"]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", name) or name in names:
            raise ValueError("Model names must be unique, 1–32 ASCII letters/digits/_/-")
        names.add(name)
        count = target.get("routes", 1)
        if type(count) is not int or count < 1:
            raise ValueError("routes must be a positive integer")
        target["routes"] = count
        target["warmup_payload"] = resolve_path(target["warmup_payload"], path.parent)
        if not Path(target["warmup_payload"]).is_file() or not Path(target["warmup_payload"]).stat().st_size:
            raise ValueError("warmup_payload must be a nonempty captured request in binary wire format")
        if target.get("factory"):
            module, separator, function = target["factory"].rpartition(":")
            if not separator or not module or not function:
                raise ValueError("factory must be module:function or /path/adapter.py:function")
            if module.endswith(".py"):
                target["factory"] = resolve_path(module, path.parent) + ":" + function
        else:
            checked_url(target["endpoint"], ("ws", "wss"))
        if target.get("validator", "corobot") not in ("corobot", "bytes"):
            raise ValueError("validator must be corobot or bytes")
        if "action_horizon" in target and (type(target["action_horizon"]) is not int or target["action_horizon"] < 1):
            raise ValueError("action_horizon must be a positive integer")
        for key, default in (("response_timeout", 90), ("connect_retry_seconds", 10)):
            value = target.setdefault(key, default)
            if not isinstance(value, (int, float)) or not 0 < value <= 100:
                raise ValueError(key + " must be in (0, 100] seconds")
    for key in ("state_file", "state_dir", "auth_file"):
        if config.get(key):
            config[key] = resolve_path(config[key], path.parent)
    return config


def validate_response(raw, target):
    if not isinstance(raw, bytes) or not raw:
        raise ValueError("Inference must return a nonempty binary response")
    if target.get("validator", "corobot") == "bytes":
        return
    envelope = msgpack.unpackb(raw, raw=False)
    if not isinstance(envelope, dict) or envelope.get("error"):
        raise ValueError("Upstream rejected request or returned a non-CoRobot envelope")
    result = envelope.get("result")
    if not isinstance(result, dict):
        raise ValueError("Missing CoRobot result")
    def check_chunk(values):
        if not isinstance(values, list) or not values or not isinstance(values[0], list) or not values[0]:
            raise ValueError("Action chunk must be a nonempty 2D array")
        width = len(values[0])
        for row in values:
            if not isinstance(row, list) or len(row) != width or any(
                type(value) not in (int, float) or not math.isfinite(value) for value in row
            ):
                raise ValueError("Action contains ragged/nonfinite/non-numeric values")
        return len(values)

    lengths = []
    for key in ("left_arm", "right_arm", "left_effector", "right_effector"):
        values = result[key]
        if key.endswith("arm"):
            if values.get("kind") not in ("JOINT_ABS", "EEF_ABS"):
                raise ValueError("Unsupported CoRobot arm action kind")
            values = values["values"]
        lengths.append(check_chunk(values))
    for key in ("waist", "wheels", "head"):
        if key in result and result[key] is not None:
            values = result[key]
            lengths.append(check_chunk(values.get("values") if isinstance(values, dict) else values))
    if len(set(lengths)) != 1:
        raise ValueError("Action chunks have inconsistent horizons")
    if target.get("action_horizon") is not None and lengths[0] != target["action_horizon"]:
        raise ValueError("Unexpected action horizon")


class WebSocketBackend:
    def __init__(self, target):
        self.target = target
        self.ws = None

    def __call__(self, session_id, payload):
        if self.ws is None:
            deadline = time.monotonic() + self.target["connect_retry_seconds"]
            while True:
                try:
                    self.ws = connect(self.target["endpoint"], compression=None, max_size=None,
                                      ping_interval=None, open_timeout=min(5, self.target["connect_retry_seconds"]))
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(min(.25, max(0, deadline - time.monotonic())))
        # Once a request might have reached the model, do not silently replay it.
        try:
            self.ws.send(payload)
            return self.ws.recv(timeout=self.target["response_timeout"])
        except Exception:
            self.close()
            raise

    def close(self):
        if self.ws is not None:
            self.ws.close()
            self.ws = None


class Relay:
    def __init__(self, target):
        self.target = target
        self.fixture = Path(target["warmup_payload"]).read_bytes()
        self.lock = threading.RLock()
        self.warmed = False
        if not target.get("factory"):
            self.backend = WebSocketBackend(target)
        else:
            module, _, function = target["factory"].rpartition(":")
            if module.endswith(".py"):
                spec = importlib.util.spec_from_file_location("route_adapter_" + uuid.uuid4().hex, module)
                imported = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = imported
                spec.loader.exec_module(imported)
            else:
                imported = importlib.import_module(module)
            self.backend = getattr(imported, function)(target)

    def warmup(self):
        with self.lock:
            if not self.warmed:
                validate_response(self.backend("__route_probe__", self.fixture), self.target)
                callback = getattr(self.backend, "on_session_close", None)
                if callback:
                    callback("__route_probe__")
                self.warmed = True

    def __call__(self, session_id, payload):
        with self.lock:
            if not payload:
                self.warmup()
                return b""  # Tunnel warmup no-op only; never synthesize an action for real data.
            response = self.backend(session_id, payload)
            validate_response(response, self.target)
            return response  # Byte-for-byte, including waist, base and unknown future fields.

    def on_session_close(self, session_id):
        callback = getattr(self.backend, "on_session_close", None)
        if callback:
            with self.lock:
                callback(session_id)

    def close(self):
        callback = getattr(self.backend, "close", None)
        if callback:
            callback()


def desired_slots(config):
    return [(f"route_{target['name']}_{index}", target)
            for target in config["models"] for index in range(target["routes"])]


def backend_config(target):
    return {key: value for key, value in target.items() if key != "routes"}


def probe(config):
    def one(pair):
        slot, target = pair
        relay = Relay(target)
        started = time.monotonic()
        try:
            relay.warmup()
            return {"slot": slot, "model": target["name"], "passed": True,
                    "seconds": round(time.monotonic() - started, 4)}
        finally:
            relay.close()
    pairs = desired_slots(config)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(pairs)) as pool:
        return list(pool.map(one, pairs))


def bind_job(config):
    state = read_state(config["state_file"])
    auth = read_state(config["auth_file"]) if config.get("auth_file") else state
    base_url = config.get("api_base") or state.get("BASE_URL") or auth.get("BASE_URL")
    if not base_url:
        raise ValueError("API host unknown: set api_base in the route config or BASE_URL in the state file")
    base = checked_url(base_url, ("http", "https"))
    # State files from challenge-login/submit-job carry no BASE_URL; only a recorded host can mismatch.
    if state.get("BASE_URL") and state["BASE_URL"].rstrip("/") != base:
        raise ValueError("Job state/API host mismatch")
    if auth.get("BASE_URL") and auth["BASE_URL"].rstrip("/") != base:
        raise ValueError("Login state/API host mismatch")
    job_id = int(config.get("job_id", state.get("JOB_ID", 0)))
    job_uuid = str(uuid.UUID(config.get("job_uuid", state.get("JOB_UUID", ""))))
    if state.get("JOB_TOKEN") and (state.get("JOB_UUID") != job_uuid or int(state.get("JOB_ID", 0)) != job_id):
        raise ValueError("Saved JOB_TOKEN belongs to another job")
    job = get_job(base, auth.get("CHALLENGE_TOKEN"), job_id, config.get("season") or state.get("JOB_SEASON"))
    if job["uuid"] != job_uuid:
        raise ValueError("JOB_ID/JOB_UUID do not match the platform")
    active_job(job)
    profile = api(base, "/api/challenge/current-user-info", auth.get("CHALLENGE_TOKEN"))
    caps = [int(value) for value in (state.get("PARALLELISM"), profile.get("max_concurrent_cases")) if value is not None]
    if not caps or min(caps) < 1 or len(desired_slots(config)) > min(caps):
        raise ValueError("Requested routes exceed known current concurrency cap, or cap is unavailable")
    gateway = checked_url(config.get("gateway_url") or state.get("TUNNEL_ENDPOINT", ""), ("ws", "wss"))
    config.update(_job_id=job_id, _job_uuid=job_uuid, _gateway=gateway)
    return job


def worker(config_path, run_dir, slot):
    config = json.loads(Path(config_path).read_text())
    target = dict(desired_slots(config))[slot]
    state = read_state(config["state_file"])
    auth = read_state(config["auth_file"]) if config.get("auth_file") else state
    if state.get("JOB_TOKEN") and state.get("JOB_UUID") != config["_job_uuid"]:
        raise ValueError("Job credential changed to another job")
    relay = Relay(target)
    health = {"slot": slot, "pid": os.getpid(), "phase": "STARTING", "gateway_connected": False,
              "actions_generated": 0, "handler_errors": 0, "started_at": now()}
    health_lock = threading.RLock()
    health_path = Path(run_dir) / (slot + ".json")

    def persist(**values):
        with health_lock:
            health.update(values)
            health["updated_at"] = now()
            atomic_json(health_path, health)

    def handle(sid, payload):
        started = time.monotonic()
        if payload and not health.get("drained"):
            persist(gateway_connected=True, phase="RUNNING", last_observation_at=now())
        try:
            response = relay(sid, payload)
        except Exception as exc:
            persist(handler_errors=health["handler_errors"] + 1, last_error_at=now(), last_error_type=type(exc).__name__)
            raise
        if payload:
            with health_lock:
                count = health["actions_generated"] + 1
                persist(actions_generated=count, last_action_at=now(), last_session_id=sid,
                        last_inference_seconds=round(time.monotonic() - started, 4))
            if count <= 3 or count % 50 == 0:
                LOG.info("action ready slot=%s session=%s request=%d seconds=%.3f", slot, sid, count,
                         time.monotonic() - started)
        return response

    persist()
    exit_code = 1
    try:
        relay.warmup()  # Actual model request; a listening port is not readiness.
        persist(model_warmed_at=now(), phase="DIALING")
        state_dir = Path(config.get("state_dir", str(Path.home() / ".simubotix")))
        agent_id = load_or_create_agent_id(config["_job_uuid"], slot, state_dir)
        persist(agent_id=agent_id)
        def connected():
            # A reattach within the reconnect window gets no control frame, and an idle
            # route gets no data either, so a tunnel that stays open is the readiness signal.
            with health_lock:
                if health.get("phase") in ("DIALING", "RECONNECTING", "STARTING"):
                    persist(gateway_connected=True, phase="CONNECTED", last_connected_at=now())
                else:
                    persist(gateway_connected=True, last_connected_at=now())

        client = TunnelClient(config["_gateway"], state.get("JOB_TOKEN") or auth.get("CHALLENGE_TOKEN"),
                              config["_job_uuid"], handle, agent_id=agent_id,
                              on_session_close=relay.on_session_close, on_connected=connected)
        original = client._handle_control
        original_run_once = client.run_once

        async def control(ws, message):
            try:
                event = json.loads(message)
            except (ValueError, TypeError):
                return
            if not isinstance(event, dict):
                return
            await original(ws, message)
            kind = event.get("type")
            values = {"gateway_connected": True, "last_control_at": now()}
            if kind == "queued":
                values.update(phase="QUEUED", queue_reason=event.get("reason"))
            elif kind == "drain":
                values.update(phase="DRAINING", drained=True)
            elif kind in ("warmup", "session_open", "session_close"):
                values.update(phase=client.state.value, active_sessions=sorted(client.active_sessions))
            persist(**values)

        async def run_once():
            try:
                await original_run_once()
            finally:
                if not health.get("drained"):
                    persist(gateway_connected=False, phase="RECONNECTING")

        client._handle_control = control
        client.run_once = run_once
        asyncio.run(client.run())
        persist(phase="DRAINED", gateway_connected=False, drained=True)
        exit_code = 0
    except KeyboardInterrupt:
        persist(phase="STOPPED", gateway_connected=False)
        exit_code = 130
    except TunnelRejected:
        persist(phase="REJECTED", gateway_connected=False)
        LOG.exception("Gateway refused this route; check token/job/cap")
        exit_code = 2
    except Exception:
        persist(phase="FAILED", gateway_connected=False)
        LOG.exception("Route failed")
    finally:
        relay.close()
        persist(exit_code=exit_code)
    return exit_code


def status(run_dir):
    path = Path(run_dir) / "routes.json"
    manifest = json.loads(path.read_text())
    rows = []
    for record in manifest["routes"]:
        health_path = Path(run_dir) / (record["slot"] + ".json")
        health = json.loads(health_path.read_text()) if health_path.exists() else {}
        live = owns_process(record)
        rows.append({"slot": record["slot"], "pid": record["pid"], "process_alive": live,
                     "log": record["log"], "health": health})
    return {"job_id": manifest["job_id"], "job_uuid": manifest["job_uuid"], "routes": rows,
            "processes_alive": sum(row["process_alive"] for row in rows)}


def stop_records(records, timeout=8):
    for record in records:
        if owns_process(record):
            os.kill(record["pid"], signal.SIGTERM)
        elif process_identity(record["pid"]) is not None:
            raise ValueError("PID identity changed; refusing to stop an unrelated process")
    deadline = time.monotonic() + timeout
    while any(owns_process(record) for record in records) and time.monotonic() < deadline:
        time.sleep(.1)
    if any(owns_process(record) for record in records):
        raise RuntimeError("Some routes did not stop; inspect logs/processes before escalating")


def start(config_path, run_dir, startup_timeout):
    config = load_config(config_path)
    job = bind_job(config)  # Read-only fresh platform state, before creating any process.
    run_dir = Path(run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(run_dir, 0o700)
    with locked(run_dir / ".routes.lock"):
        path = run_dir / "routes.json"
        old = json.loads(path.read_text()) if path.exists() else {"routes": []}
        if old.get("job_uuid", config["_job_uuid"]) != config["_job_uuid"]:
            raise ValueError("run-dir belongs to another job; use a separate directory")
        desired = dict(desired_slots(config))
        previous = {record["slot"]: record for record in old["routes"]}
        for record in old["routes"]:
            if owns_process(record) and (record["slot"] not in desired or
                                        backend_config(record["target"]) != backend_config(desired[record["slot"]])):
                raise ValueError("A live route would change/disappear. Stop that slot explicitly first")
        pending = []
        for slot, target in desired.items():
            record = previous.get(slot)
            if record and owns_process(record):
                continue
            hpath = run_dir / (slot + ".json")
            if hpath.exists() and json.loads(hpath.read_text()).get("drained"):
                raise ValueError("Route received drain; do not reconnect this job")
            pending.append((slot, target))
        if not pending:
            atomic_json(run_dir / "config.json", config)
            atomic_json(path, {"job_id": job["id"], "job_uuid": job["uuid"],
                               "routes": [previous[slot] for slot in desired], "updated_at": now()})
            print(json.dumps(status(run_dir), ensure_ascii=False))
            return
        # Explicit worker jobs allow the same target to appear repeatedly here.
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(pending)) as pool:
            def warm(pair):
                relay = Relay(pair[1])
                try:
                    relay.warmup()
                finally:
                    relay.close()
            list(pool.map(warm, pending))
        atomic_json(run_dir / "preflight.json", {"passed": True, "routes_checked": len(pending), "checked_at": now()})
        snapshot = run_dir / "config.json"
        atomic_json(snapshot, config)
        records = [previous[slot] for slot in desired if slot in previous and owns_process(previous[slot])]
        added = []
        state_dir = Path(config.get("state_dir", str(Path.home() / ".simubotix")))
        state_dir.mkdir(parents=True, exist_ok=True)
        try:
            for slot, target in pending:
                agent_id = load_or_create_agent_id(config["_job_uuid"], slot, state_dir)
                identity_file = state_dir / f"agent_{config['_job_uuid']}_{slot}.id"
                if not identity_file.is_file() or identity_file.read_text().strip() != agent_id:
                    raise ValueError("Cannot persist agent identity; fix state_dir before connecting")
                atomic_json(run_dir / (slot + ".json"), {"phase": "STARTING", "actions_generated": 0})
                log = run_dir / (slot + ".log")
                args = [sys.executable, str(SCRIPT), "worker", "--config", str(snapshot),
                        "--run-dir", str(run_dir), "--slot", slot]
                with log.open("ab") as output:
                    process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=output,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                # /proc can be observed during exec. Persist only the verified worker image,
                # never a transient/empty argv which would later look like PID reuse.
                identity_deadline = time.monotonic() + 2
                identity = None
                while time.monotonic() < identity_deadline:
                    identity = process_identity(process.pid)
                    if identity and identity["cmdline"] == args:
                        break
                    if process.poll() is not None:
                        break
                    time.sleep(.01)
                if not identity or identity["cmdline"] != args:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=3)
                    raise RuntimeError("New worker exited or could not establish a stable process identity")
                record = {"slot": slot, "pid": process.pid, "identity": identity,
                          "agent_id": agent_id, "log": str(log), "target": target}
                added.append(record)
                records.append(record)
                atomic_json(path, {"job_id": job["id"], "job_uuid": job["uuid"], "routes": records, "updated_at": now()})
            deadline = time.monotonic() + startup_timeout
            while time.monotonic() < deadline:
                rows = status(run_dir)["routes"]
                if any(row["health"].get("phase") in ("FAILED", "REJECTED", "STOPPED") for row in rows):
                    raise RuntimeError("Worker failed; inspect the slot log")
                if any(row["health"].get("drained") for row in rows):
                    break  # Normal completion can race launch; never relaunch after drain.
                if all(row["process_alive"] and row["health"].get("gateway_connected") and
                       row["health"].get("phase") in ("CONNECTED", "RUNNING", "QUEUED") for row in rows):
                    break
                time.sleep(.2)
            else:
                raise TimeoutError("Gateway readiness timeout; inspect logs, cap and job state")
        except BaseException:
            stop_records(added)
            raise
    print(json.dumps(status(run_dir), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("probe", "start", "status", "stop", "worker"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--slot", action="append")
    parser.add_argument("--startup-timeout", type=float, default=45)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.action == "probe":
        if not args.config:
            parser.error("probe requires --config")
        print(json.dumps({"passed": True, "checks": probe(load_config(args.config))}))
    elif args.action == "start":
        if not (args.config and args.run_dir):
            parser.error("start requires --config and --run-dir")
        start(args.config, args.run_dir, args.startup_timeout)
    elif args.action == "worker":
        if not (args.config and args.run_dir and args.slot and len(args.slot) == 1):
            parser.error("worker requires config, run-dir and one slot")
        raise SystemExit(worker(args.config, args.run_dir, args.slot[0]))
    else:
        if not args.run_dir:
            parser.error("status/stop requires --run-dir")
        if args.action == "stop":
            with locked(args.run_dir / ".routes.lock"):
                manifest = json.loads((args.run_dir / "routes.json").read_text())
                records = [r for r in manifest["routes"] if not args.slot or r["slot"] in args.slot]
                if args.slot and set(args.slot) - {r["slot"] for r in records}:
                    raise ValueError("Unknown slot; no process stopped")
                stop_records(records)
                for record in records:
                    health_path = args.run_dir / (record["slot"] + ".json")
                    health = json.loads(health_path.read_text()) if health_path.exists() else {}
                    health.update(phase="STOPPED", gateway_connected=False, stopped_by_manager=True, stopped_at=now())
                    atomic_json(health_path, health)
        print(json.dumps(status(args.run_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()

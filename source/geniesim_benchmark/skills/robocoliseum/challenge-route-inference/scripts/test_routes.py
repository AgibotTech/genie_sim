# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Isolated real-socket tests. Never contacts the official platform or a real model."""
import asyncio
import contextlib
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.parse
import uuid

import msgpack
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve
import tunnel_client

from common import atomic_json, process_identity, read_state, write_state
from route_inference import Relay, bind_job, validate_response
from tunnel_client import TunnelClient, decode_data_frame, encode_data_frame

HERE = Path(__file__).resolve().parent
JOB_UUID = "04113815-2212-4b20-909b-903577ebc147"
TOKEN = "mock-private-credential-never-print"


def response(wire):
    result = {"left_arm": {"kind": "JOINT_ABS", "values": [[.1, .2], [.3, .4]]},
              "right_arm": {"kind": "JOINT_ABS", "values": [[.5, .6], [.7, .8]]},
              "left_effector": [[0.0], [0.0]], "right_effector": [[0.0], [0.0]],
              "waist": [[0] * 5] * 2, "wheels": [[0] * 3] * 2,
              "preserve_unknown_fields": {"digest": hashlib.sha256(wire).hexdigest()}}
    return msgpack.packb({"result": result}, use_bin_type=True)


def wait_for(predicate, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.05)
    raise AssertionError("Timed out waiting for observable condition")


class FakePlatform:
    def __init__(self):
        self.ready = True
        self.acked = set()
        self.ack_counts = {}
        self.connections = {}
        self.lock = threading.Lock()
        self.drain = threading.Event()
        self.errors = []
        self.last_wire = None
        self.job_season = "2"

    def model(self, ws):
        try:
            for wire in ws:
                self.last_wire = wire
                ws.send(response(wire) if self.ready else b"not-ready")
        except ConnectionClosed:
            pass

    def gateway(self, ws):
        try:
            if ws.request.headers.get("Authorization") != "Bearer " + TOKEN:
                raise AssertionError("Wrong/missing tunnel credential")
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(ws.request.path).query)
            agent = query["agent"][0]
            if query["job"][0] != JOB_UUID:
                raise AssertionError("Wrong job UUID")
            with self.lock:
                reconnect = agent in self.connections
                self.connections[agent] = self.connections.get(agent, 0) + 1
            if not reconnect:
                ws.send(json.dumps({"type": "unknown_future_control"}))
                ws.send(json.dumps({"type": "warmup"}))
                if json.loads(ws.recv(timeout=10))["type"] != "ready":
                    raise AssertionError("No ready after warmup")
                ws.send(json.dumps({"type": "session_open", "session_id": agent}))
            # Reconnect deliberately sends binary data without a warmup/control replay.
            wire = b"\x00actual-platform-payload\xff"
            ws.send(encode_data_frame(agent, wire))
            sid, actual = decode_data_frame(ws.recv(timeout=10))
            if sid != agent or actual != response(wire):
                raise AssertionError("Tunnel altered the session id, request or complete response")
            with self.lock:
                self.acked.add(agent)
                self.ack_counts[agent] = self.ack_counts.get(agent, 0) + 1
            while not self.drain.is_set():
                try:
                    ws.recv(timeout=.1)
                except TimeoutError:
                    continue
            ws.send(json.dumps({"type": "drain"}))
        except ConnectionClosed:
            pass
        except Exception as exc:
            self.errors.append(repr(exc))

    def http_handler(self):
        platform = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, value):
                data = json.dumps(value).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.headers.get("Authorization") != "Bearer " + TOKEN:
                    self.send_error(401)
                elif self.path.startswith("/api/challenge/current-user-info"):
                    self.reply({"name": "mock", "model_name_approved": True, "model_type": "policy",
                                "max_concurrent_cases": 8})
                elif self.path.startswith("/api/challenge/jobs"):
                    # Like the platform: one season per listing, Season II when omitted.
                    query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
                    season = query.get("season", ["2"])[0]
                    items = [{"id": 7, "uuid": JOB_UUID, "status": 1, "detailed_status": "evaluating",
                              "progress": {"done": 0, "total": 2}}] if season == platform.job_season else []
                    self.reply({"items": items, "total_pages": 1})
                else:
                    self.send_error(404)

        return Handler


class RouteTests(unittest.TestCase):
    def run_script(self, script, *args, ok=True):
        if args and args[0] == "start" and "--startup-timeout" not in args:
            args = (*args, "--startup-timeout", "12")
        result = subprocess.run([sys.executable, str(HERE / script), *map(str, args)],
                                capture_output=True, text=True, timeout=45)
        if ok:
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        else:
            self.assertNotEqual(result.returncode, 0)
        self.assertNotIn(TOKEN, result.stdout + result.stderr)
        return result

    def test_validation_and_custom_adapter(self):
        target = {"action_horizon": 2}
        validate_response(response(b"x"), target)
        bad = msgpack.unpackb(response(b"x"), raw=False)
        bad["result"]["wheels"][0][0] = float("nan")
        with self.assertRaises(ValueError):
            validate_response(msgpack.packb(bad), target)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = root / "probe.bin"
            fixture.write_bytes(b"probe")
            module = root / "adapter.py"
            module.write_text("class Adapter:\n def __call__(self, sid, wire): return b'R:' + wire\n"
                              "def create_handler(target): return Adapter()\n")
            relay = Relay({"factory": str(module) + ":create_handler", "validator": "bytes",
                           "warmup_payload": str(fixture)})
            relay.warmup()
            self.assertEqual(relay("s", b"x"), b"R:x")
            self.assertEqual(relay("", b""), b"")
            relay.close()

    def test_end_to_end_scale_resume_and_drain(self):
        platform = FakePlatform()
        with tempfile.TemporaryDirectory(prefix="route skill test ") as tmp, contextlib.ExitStack() as stack:
            root = Path(tmp)
            models = [stack.enter_context(serve(platform.model, "127.0.0.1", 0)) for _ in range(2)]
            gateway = stack.enter_context(serve(platform.gateway, "127.0.0.1", 0))
            for server in [*models, gateway]:
                threading.Thread(target=server.serve_forever, daemon=True).start()
            http = ThreadingHTTPServer(("127.0.0.1", 0), platform.http_handler())
            threading.Thread(target=http.serve_forever, daemon=True).start()
            stack.callback(http.server_close)
            stack.callback(http.shutdown)
            state_path = root / "job.env"
            write_state(state_path, {"BASE_URL": f"http://127.0.0.1:{http.server_port}", "CHALLENGE_TOKEN": TOKEN,
                        "JOB_ID": 7, "JOB_UUID": JOB_UUID, "JOB_TOKEN": TOKEN, "PARALLELISM": 8,
                        "TUNNEL_ENDPOINT": f"ws://127.0.0.1:{gateway.socket.getsockname()[1]}/api/challenge/tunnel"})
            auth_path = root / "login.env"
            write_state(auth_path, {"BASE_URL": f"http://127.0.0.1:{http.server_port}", "CHALLENGE_TOKEN": TOKEN})
            state = read_state(state_path)
            state["CHALLENGE_TOKEN"] = "expired-login-token"
            write_state(state_path, state)
            fixture = root / "probe.bin"
            fixture.write_bytes(b"probe-wire")
            config = {"state_file": str(state_path), "auth_file": str(auth_path), "state_dir": str(root / "ids"), "models": [
                {"name": f"model{i}", "endpoint": f"ws://127.0.0.1:{s.socket.getsockname()[1]}",
                 "routes": 1, "warmup_payload": str(fixture), "action_horizon": 2} for i, s in enumerate(models)]}
            config_path = root / "routes.json"
            atomic_json(config_path, config)
            run_dir = root / "run"
            try:
                platform.ready = False
                self.run_script("route_inference.py", "start", "--config", config_path, "--run-dir", run_dir, ok=False)
                self.assertFalse((run_dir / "routes.json").exists())
                platform.ready = True
                self.run_script("route_inference.py", "start", "--config", config_path, "--run-dir", run_dir)
                first = json.loads((run_dir / "routes.json").read_text())
                first_pids = {r["pid"] for r in first["routes"]}
                for m in config["models"]:
                    m["routes"] = 4
                atomic_json(config_path, config)
                self.run_script("route_inference.py", "start", "--config", config_path, "--run-dir", run_dir)
                wait_for(lambda: len(platform.acked) == 8)
                manifest = json.loads((run_dir / "routes.json").read_text())
                self.assertTrue(first_pids.issubset({r["pid"] for r in manifest["routes"]}))
                self.assertEqual(len({r["agent_id"] for r in manifest["routes"]}), 8)
                self.run_script("route_inference.py", "start", "--config", config_path, "--run-dir", run_dir)
                repeated = json.loads((run_dir / "routes.json").read_text())
                self.assertEqual({r["slot"]: r["pid"] for r in manifest["routes"]},
                                 {r["slot"]: r["pid"] for r in repeated["routes"]})
                slot = manifest["routes"][0]["slot"]
                identity = manifest["routes"][0]["agent_id"]
                self.run_script("route_inference.py", "stop", "--run-dir", run_dir, "--slot", slot)
                self.run_script("route_inference.py", "start", "--config", config_path, "--run-dir", run_dir)
                wait_for(lambda: platform.ack_counts.get(identity) == 2)
                status = json.loads(self.run_script("route_inference.py", "status", "--run-dir", run_dir).stdout)
                self.assertEqual(status["processes_alive"], 8)
                self.assertTrue(all(r["health"].get("actions_generated", 0) >= 1 for r in status["routes"]))
                self.assertEqual(platform.errors, [])
                for log in run_dir.glob("*.log"):
                    self.assertNotIn(TOKEN, log.read_text())
                config["models"][0]["routes"] = 5
                atomic_json(config_path, config)
                self.run_script("route_inference.py", "start", "--config", config_path, "--run-dir", run_dir, ok=False)
                config["models"][0]["routes"] = 4
                atomic_json(config_path, config)
                platform.drain.set()
                wait_for(lambda: all(not process_identity(r["pid"]) for r in
                                    json.loads((run_dir / "routes.json").read_text())["routes"]))
                self.run_script("route_inference.py", "start", "--config", config_path, "--run-dir", run_dir, ok=False)
            finally:
                if (run_dir / "routes.json").exists():
                    if sys.exc_info()[0] is not None:
                        for record in json.loads((run_dir / "routes.json").read_text())["routes"]:
                            current = process_identity(record["pid"])
                            if current != record["identity"]:
                                print("MOCK_IDENTITY_MISMATCH", record["slot"], record["identity"], current, file=sys.stderr)
                            health = run_dir / (record["slot"] + ".json")
                            if health.exists():
                                print("MOCK_HEALTH", health.read_text(), file=sys.stderr)
                        print("MOCK_GATEWAY_ERRORS", platform.errors, file=sys.stderr)
                    self.run_script("route_inference.py", "stop", "--run-dir", run_dir)

    def test_season_one_job_and_state_without_base_url(self):
        platform = FakePlatform()
        platform.job_season = "1"
        http = ThreadingHTTPServer(("127.0.0.1", 0), platform.http_handler())
        threading.Thread(target=http.serve_forever, daemon=True).start()
        try:
            base = f"http://127.0.0.1:{http.server_port}"
            gateway = "ws://127.0.0.1:1/api/challenge/tunnel"
            with tempfile.TemporaryDirectory() as tmp:
                state = Path(tmp) / "challenge.env"

                def bind(values, **config):
                    write_state(state, values)
                    config = {"state_file": str(state), "models": [{"name": "m", "routes": 1}], **config}
                    bind_job(config)
                    return config

                # Layout challenge-login/challenge-submit-job write: no BASE_URL, JOB_SEASON=1.
                job = {"CHALLENGE_TOKEN": TOKEN, "JOB_ID": 7, "JOB_UUID": JOB_UUID, "PARALLELISM": 2,
                       "TUNNEL_ENDPOINT": gateway, "JOB_TOKEN": TOKEN}
                config = bind({**job, "JOB_SEASON": 1}, api_base=base)
                self.assertEqual((config["_job_id"], config["_gateway"]), (7, gateway))
                # No season recorded: Season II is searched first, then Season I.
                bind({**job, "BASE_URL": base})
                # No API host anywhere is an explicit error, not a guess.
                with self.assertRaises(ValueError):
                    bind(job)
                # A recorded host that differs is still rejected.
                with self.assertRaises(ValueError):
                    bind({**job, "BASE_URL": "http://127.0.0.1:1"}, api_base=base)
        finally:
            http.shutdown()
            http.server_close()

    def test_connected_signal_without_control_frames(self):
        # A reattached or not-yet-scheduled agent receives no control frame at all.
        hold = threading.Event()

        def silent_gateway(ws):
            hold.wait(10)

        def rejecting_gateway(ws):
            ws.close()  # what the gateway does to a dial over the parallelism cap

        async def run(handler):
            seen = []
            with serve(handler, "127.0.0.1", 0) as server:
                threading.Thread(target=server.serve_forever, daemon=True).start()
                url = f"ws://127.0.0.1:{server.socket.getsockname()[1]}/api/challenge/tunnel"
                client = TunnelClient(url, TOKEN, JOB_UUID, lambda sid, payload: b"",
                                      on_connected=lambda: seen.append(True))
                task = asyncio.create_task(client.run_once())
                await asyncio.sleep(tunnel_client.CONNECTED_GRACE_SECONDS + 1)
                task.cancel()
                with contextlib.suppress(BaseException):
                    await task
                hold.set()
                server.shutdown()
            return seen

        self.assertEqual(asyncio.run(run(silent_gateway)), [True])
        self.assertEqual(asyncio.run(run(rejecting_gateway)), [])

    def test_stop_refuses_reused_or_unrelated_pid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            process = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(45)"])
            try:
                identity = process_identity(process.pid)
                identity["start_ticks"] = "not-this-process"
                atomic_json(root / "routes.json", {"job_id": 7, "job_uuid": JOB_UUID, "routes": [
                    {"pid": process.pid, "identity": identity, "slot": "wrong", "log": "unused"}]})
                self.run_script("route_inference.py", "stop", "--run-dir", root, ok=False)
                self.assertIsNone(process.poll())
            finally:
                process.terminate()
                process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)

# Adapt existing inference services

For application payload fields and backend-specific behavior, read
[the simulator inference protocol](protocol.md).

## Transparent WebSocket mode

The tunnel wraps each payload as a big-endian uint32 session-ID byte length,
UTF-8 session ID, then payload bytes. The bundled client handles this framing,
authorization, keepalive, and reconnect. The relay forwards only payload bytes to
the existing service and returns its complete binary response.

The default service accepts a CoRobot MessagePack `method` / `params` request and
returns a `result` envelope. Preserve all fields, including embodiment tags,
depth, history, EEF frames, chassis targets, and terminal metadata. Do not infer
`ee_type` from `robot_type` alone or flatten every action into 16 dimensions.
Widths and horizon come from the actual embodiment and model.

`warmup_payload` must contain a real request in the service's accepted wire format,
without tunnel framing. Export it with the existing client's encoder. Do not use
JSON text for binary images or silently change NumPy/extension serialization.
This skill does not load pickle files. Export trusted `.pkl` observations into
actual request bytes using the originating tool before handing them to the relay.

An empty tunnel warmup payload triggers a real fixture readiness check, followed
by a `b""` no-op. Nonempty request failures are reported; fixture actions never
replace real inference responses. Connection establishment may retry within
`connect_retry_seconds`; a possibly sent request is not automatically replayed,
because replay can advance a stateful model twice. The default
`response_timeout` is 90 seconds; include network transfer and queueing when
comparing it with the runtime's complete inference timeout.

## HTTP, RPC, native OpenPI, and stateful services

Reuse the user's working simulation encoding, camera mapping, action conversion,
and reset logic. A trusted custom adapter can be configured as follows:

```json
{
  "name": "custom_model",
  "routes": 2,
  "factory": "./my_adapter.py:create_handler",
  "warmup_payload": "./real-platform-observation.bin",
  "validator": "corobot",
  "options": {"service_name": "use-your-existing-client-configuration"}
}
```

`create_handler(target: dict)` returns a synchronous callable with this interface:

```python
def create_handler(target):
    # Reuse a verified client; read authentication from private configuration.
    return ExistingSimulationAdapter(target.get("options", {}))

class ExistingSimulationAdapter:
    def __init__(self, options):
        ...  # Create an independent client for this route.

    def __call__(self, session_id: str, payload: bytes) -> bytes:
        ...  # Decode, infer, and encode the simulator-compatible response.

    def on_session_close(self, session_id: str):
        ...  # Optional: release this session's history and caches.

    def close(self):
        ...  # Optional: close this route's client connection.
```

This is an interface sketch, not an executable adapter. Point `factory` at a real,
trusted implementation. Importable `package.module:create_handler` references
also work. The code executes in each route process.

Probes use the reserved session ID `__route_probe__` and then call its close hook.
Keep that temporary state isolated from real sessions. Explicitly implement any
server greeting, HTTP JSON, native OpenPI metadata, or compression; the relay does
not guess a protocol or discard an unsolicited first frame.

The bundled `corobot` validator requires ordinary action chunks. It does not
accept the simulator's special no-action terminal response or native Newton's
terminal-only reply. If those are needed, provide an adapter that validates the
full target-runtime contract and configure `validator: "bytes"`. That setting
only checks nonempty binary data; it delegates semantic validation to the adapter
and does not establish that arbitrary bytes are valid actions.

## Shared models and identities

Each route is an OS process with a persistent `agent_id` and independent backend
client. Its handler serializes requests within that route, but a shared model can
receive requests from several routes concurrently. Independent connections and
thread locks do not isolate global episode history, RNG state, or prompt caches.
Use session-aware service/adapter state or separate model processes, and probe
interleaved requests rather than checking shapes alone.

The existing model launcher controls GPU placement and memory. This skill does
not load the model or choose its GPU index. Determine routes per service from
actual concurrency, latency, and memory measurements. A model, weights, port, or
memory-pool change invalidates earlier readiness checks.

## Bundled tunnel client

`scripts/tunnel_client.py` implements the tunnel protocol described in the
[run-agent appendix](../../challenge-run-agent/SKILL.md#appendix--tunnel-protocol-for-a-self-built-client);
`scripts/agent_identity.py` persists each route's `agent_id`. They have no pi05,
LingBot, or JAX dependency; they use the Python standard library and websockets.
`scripts/test_routes.py` holds the isolated mock regression tests.

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""A synchronized camera response is produced only by a completed Replicator batch."""

from __future__ import annotations

import base64
from collections import deque
import math
from fractions import Fraction
import io
import json
import time

import numpy as np


def _reference_stamp(value):
    numerator, denominator = value
    if (
        any(
            not isinstance(item, (int, np.integer)) or isinstance(item, (bool, np.bool_))
            for item in (numerator, denominator)
        )
        or numerator < 0
        or denominator <= 0
    ):
        raise ValueError("invalid rational render reference metadata")
    return Fraction(int(numerator), int(denominator))


class CaptureBatch:
    """Own all RGB arrays from one Writer callback and reject old render frames."""

    def __init__(self, products: dict[str, str], resolutions=None):
        self.products = dict(products)
        self.resolutions = dict(resolutions or {})
        self.latest_frame = -1
        self.latest_time = Fraction(-1)
        self.baseline = (-1, Fraction(-1))
        self.packet = None
        self.error = None

    def arm(self, reference_floor=None):
        reference = self.latest_time if reference_floor is None else max(self.latest_time, reference_floor)
        self.baseline = (self.latest_frame, reference)
        self.packet = None
        self.error = None

    def accept(self, data):
        try:
            frame = data["swhFrameNumber"]
            if not isinstance(frame, (int, np.integer)) or isinstance(frame, (bool, np.bool_)) or frame < 0:
                raise ValueError("invalid render frame metadata")
            frame = int(frame)
            numerator, denominator = data["reference_time"]
            stamp = _reference_stamp((numerator, denominator))
            if stamp <= self.latest_time or stamp <= self.baseline[1]:
                return
            self.latest_frame += 1
            self.latest_time = stamp
            images = {}
            for camera, product in self.products.items():
                key = "rgb-" + product
                if key not in data:
                    raise RuntimeError(f"missing camera in completed render batch: {camera}")
                image = np.asarray(data[key])
                if image.ndim != 3 or image.shape[2] not in (3, 4) or not image.size or image.dtype != np.uint8:
                    raise RuntimeError(f"invalid camera buffer: {camera}")
                if camera in self.resolutions:
                    width, height = self.resolutions[camera]
                    if image.shape[:2] != (height, width):
                        raise RuntimeError(f"camera buffer resolution differs from render product: {camera}")
                images[camera] = image[:, :, :3].copy()
            self.packet = {
                "render_frame": self.latest_frame,
                "swh_frame_number": frame,
                "render_reference_time": [int(numerator), int(denominator)],
                "images": images,
            }
        except Exception as exc:
            self.error = exc

    def result(self):
        if self.error is not None:
            raise RuntimeError(str(self.error)) from self.error
        if self.packet is None:
            raise RuntimeError(
                f"no fresh completed RGB render batch: baseline={self.baseline}, "
                f"latest={(self.latest_frame, self.latest_time)}"
            )
        return self.packet


class ReplicatorCapture:
    """Drain old work, apply a frozen scene snapshot, then wait for new GPU output."""

    def __init__(self, cameras):
        import omni.replicator.core as rep

        self.rep = rep
        self.products = {}
        self.writer = None
        try:
            rep.orchestrator.set_capture_on_play(False)
            for name, (path, resolution) in cameras.items():
                self.products[name] = rep.create.render_product(path, resolution, name="Sync_" + name)
            self.batch = CaptureBatch(
                {name: product.path.rsplit("/", 1)[-1] for name, product in self.products.items()},
                resolutions={name: resolution for name, (path, resolution) in cameras.items()},
            )
            sink = self.batch

            class SnapshotWriter(rep.Writer):
                def __init__(self):
                    self.annotators = ["rgb"]

                def write(self, data):
                    sink.accept(data)

            self.writer = SnapshotWriter()
            self.writer.attach(list(self.products.values()))
            self._pending_baseline = self.batch.latest_frame
            self._warmup_reference_floor = self.batch.latest_time
            self._warmup()
        except Exception:
            try:
                self.close()
            except Exception:
                pass  # Cleanup must not replace the original initialization failure.
            raise

    def _warmup(self):
        from omni.kit.app import get_app

        deadline = time.monotonic() + 300.0
        print("Initializing synchronous render products", flush=True)
        for _ in range(20):
            if time.monotonic() >= deadline:
                raise RuntimeError("render-product initialization exceeded the warmup budget")
            get_app().update()
        target = None
        attempt = 0

        def check_ready():
            if self.batch.error is not None:
                raise RuntimeError(str(self.batch.error)) from self.batch.error
            # A Python budget cannot interrupt a blocking Kit call; enforce it
            # at every step/wait/update boundary before accepting readiness.
            if time.monotonic() >= deadline:
                raise RuntimeError("warmup exceeded 300-second monotonic budget")
            self._warmup_reference_floor = max(self._warmup_reference_floor, self.batch.latest_time)
            return self.batch.packet is not None and self.batch.latest_time > previous_floor

        try:
            for attempt in range(1, 4):
                previous_floor = self._warmup_reference_floor
                check_ready()
                print(
                    f"sync RGB warmup attempt={attempt}/3 previous_floor={previous_floor} "
                    f"latest_reference={self.batch.latest_time}",
                    flush=True,
                )
                self.rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=True, wait_for_render=True)
                # This public orchestrator API returns scheduled interval ENDS,
                # not pending callbacks. 5.1 retains its live deque after write;
                # 6.0.1 clears it at wait, so copy/fold immediately after step.
                scheduled = list(self.rep.orchestrator.get_sim_times_to_write())
                ends = [_reference_stamp(value) for value in scheduled]
                if not ends:
                    raise RuntimeError("missing scheduled warmup reference metadata")
                target = max(ends)
                # 6.0.1 may stamp a callback inside its interval or as late
                # as end + 1e-7. Keep this floor separate from true history.
                self._warmup_reference_floor = max(self._warmup_reference_floor, target + Fraction(1, 10_000_000))
                check_ready()
                self.rep.orchestrator.wait_until_complete()
                for _ in range(10):
                    if check_ready():
                        break
                    get_app().update()
                if check_ready():
                    self._pending_baseline = None
                    print(
                        f"sync RGB warmup ready attempt={attempt}/3 target={target} "
                        f"latest_reference={self.batch.latest_time} "
                        f"startup_floor={self._warmup_reference_floor}",
                        flush=True,
                    )
                    return
            raise RuntimeError("no complete current-attempt RGB Writer callback after 3 warmup attempts")
        except Exception as exc:
            raise RuntimeError(
                f"sync RGB warmup failed: attempt={attempt}/3 target={target} "
                f"latest_reference={self.batch.latest_time}: {exc}"
            ) from exc

    def _drain_pending_capture(self):
        """Wait for the previous Writer before assigning any subsequent snapshot."""
        baseline = getattr(self, "_pending_baseline", None)
        if baseline is None:
            return
        from omni.kit.app import get_app

        for _ in range(10):
            if self.batch.latest_frame > baseline:
                if self.batch.error is not None:
                    raise RuntimeError(str(self.batch.error)) from self.batch.error
                self._pending_baseline = None
                return
            get_app().update()
        if self.batch.latest_frame > baseline and self.batch.error is None:
            self._pending_baseline = None
            return
        raise RuntimeError("previous capture Writer callback has not completed")

    def capture(self, apply_snapshot):
        if getattr(self, "_metadata_error", None) is not None:
            raise RuntimeError("previous capture lacked scheduled reference metadata") from self._metadata_error
        self.rep.orchestrator.wait_until_complete()
        self._drain_pending_capture()
        self.batch.arm(reference_floor=getattr(self, "_warmup_reference_floor", None))
        apply_snapshot()
        self._pending_baseline = self.batch.latest_frame
        # Isaac 6.2 uses an independent Fabric reference clock even when the
        # timeline is held. Neither the timeline nor any physics solver advances.
        try:
            # step() may enqueue GPU work before raising. Until both submission
            # and schedule readback succeed, the interval is unknown and this
            # capture backend must never be reused for another snapshot.
            self.rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=True, wait_for_render=True)
            scheduled = list(self.rep.orchestrator.get_sim_times_to_write())
            ends = [_reference_stamp(value) for value in scheduled]
            if not ends:
                raise RuntimeError("missing scheduled capture reference metadata")
            self._warmup_reference_floor = max(
                getattr(self, "_warmup_reference_floor", Fraction(-1)), max(ends) + Fraction(1, 10_000_000)
            )
        except Exception as exc:
            self._metadata_error = exc
            raise RuntimeError("invalid scheduled capture reference metadata") from exc
        self.rep.orchestrator.wait_until_complete()
        # Kit 6.x can deliver Writer callbacks on subsequent app updates even
        # after the orchestrator's GPU fence completes. Keep the snapshot frozen
        # while draining those callbacks; do not accept stale/partial batches.
        from omni.kit.app import get_app

        for _ in range(10):
            if self.batch.packet is not None or self.batch.error is not None:
                break
            get_app().update()
        result = self.batch.result()
        self._pending_baseline = None
        return result

    def close(self):
        error = None
        writer, self.writer = self.writer, None
        if writer is not None:
            try:
                writer.detach()
            except Exception as exc:
                error = exc
        products, self.products = self.products, {}
        for product in products.values():
            try:
                product.destroy()
            except Exception as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error


def validate_snapshot(snapshot):
    """Validate the complete plain-data ABI before scheduling any USD writes.

    Public body poses are already authored-root world xyz+wxyz poses. The raw
    SDK archive's xyzw data never enters this channel.
    """
    if not isinstance(snapshot, dict):
        raise ValueError("snapshot must be a JSON object")
    for field in ("source_id", "request_id"):
        if not isinstance(snapshot.get(field), str) or not snapshot[field]:
            raise ValueError(f"snapshot {field} must be a nonempty string")
    for field in ("epoch",):
        if type(snapshot.get(field)) is not int or snapshot[field] < 0:
            raise ValueError(f"snapshot {field} must be a nonnegative integer")
    if snapshot.get("error"):
        raise ValueError(str(snapshot["error"]))
    if type(snapshot.get("version")) is not int or snapshot["version"] != 1:
        raise ValueError("unsupported snapshot version")
    if snapshot.get("pose_schema") != "xyz+wxyz":
        raise ValueError("unsupported snapshot pose_schema; expected xyz+wxyz")
    if not isinstance(snapshot.get("snapshot_id"), str) or not snapshot["snapshot_id"]:
        raise ValueError("snapshot_id is required")
    for field in ("tick", "cloth_tick"):
        if type(snapshot.get(field)) is not int or snapshot[field] < 0:
            raise ValueError(f"invalid snapshot {field}")
    if snapshot["tick"] != snapshot["cloth_tick"]:
        raise ValueError("snapshot cloth_tick differs from body tick")

    def number(value):
        try:
            return type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            return False

    if not number(snapshot.get("sim_time")) or snapshot["sim_time"] < 0:
        raise ValueError("invalid snapshot simulation time")
    if not isinstance(snapshot.get("joints"), dict) or any(
        not isinstance(key, str) or not key or not number(value) for key, value in snapshot["joints"].items()
    ):
        raise ValueError("invalid snapshot joints")

    def path(value):
        return isinstance(value, str) and value.startswith("/") and value != "/" and "//" not in value

    if not isinstance(snapshot.get("bodies"), dict):
        raise ValueError("invalid snapshot bodies")
    for key, pose in snapshot["bodies"].items():
        if not path(key) or not isinstance(pose, list) or len(pose) != 7 or not all(map(number, pose)):
            raise ValueError(f"invalid snapshot body pose: {key}")
        norm = math.hypot(*pose[3:])
        if not math.isfinite(norm) or abs(norm - 1.0) > 1e-3:
            raise ValueError(f"invalid snapshot body quaternion: {key}")
    if not isinstance(snapshot.get("deformables"), list):
        raise ValueError("invalid snapshot deformables")
    ids, paths = set(), set()
    for mesh in snapshot["deformables"]:
        if not isinstance(mesh, dict):
            raise ValueError("invalid snapshot deformable entry")
        mesh_id, prim_path = mesh.get("mesh_id"), mesh.get("prim_path")
        if type(mesh_id) is not int or mesh_id < 0 or mesh_id in ids or not path(prim_path) or prim_path in paths:
            raise ValueError("invalid or duplicate snapshot mesh identity")
        if not isinstance(mesh.get("name"), str) or not mesh["name"]:
            raise ValueError("invalid snapshot mesh name")
        points = mesh.get("points")
        if (
            not isinstance(points, list)
            or not points
            or any(not isinstance(row, list) or len(row) != 3 or not all(map(number, row)) for row in points)
        ):
            raise ValueError("invalid snapshot deformable points")
        ids.add(mesh_id)
        paths.add(prim_path)
    # Own the entire payload, reject non-JSON extension fields and retain every
    # input float in the immutable response instead of Float32 ROS transport.
    return json.loads(json.dumps(snapshot, sort_keys=True, allow_nan=False))


class SyncObservationRenderer:
    """One renderer lifetime binds one source UUID; epochs may only increase.

    Callbacks validate/enqueue copied data. Only process() owns scene mutation
    and GPU capture. Retried IDs reuse the exact serialized response. A backend
    restart creates a new source and requires its paired renderer to restart.
    """

    def __init__(self, node, topic, cameras, apply_snapshot, *, completed_capture=None):
        from std_msgs.msg import String
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

        if not isinstance(topic, str) or not topic.startswith("/") or topic == "/" or topic.endswith("/"):
            raise ValueError("sync_observation_topic must be an absolute topic prefix")
        self.node = node
        self.apply_snapshot = apply_snapshot
        self.completed_capture = completed_capture
        self.capture_backend = self.publisher = self.subscription = None
        self._pending = deque()
        self._payloads, self._responses = {}, {}
        self.source_id = self.epoch = None
        self._closed = False
        try:
            self.capture_backend = ReplicatorCapture(cameras)
            self.publisher = node.create_publisher(String, topic + "/observation", 10)
            # Capture startup can outlast the engine's first snapshot. Request
            # its retained immutable state; normal identity checks still reject
            # foreign sources, older epochs and conflicting request payloads.
            qos = QoSProfile(
                depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE
            )
            self.subscription = node.create_subscription(String, topic + "/snapshot", self.receive, qos)
        except BaseException:
            try:
                self.close()
            except Exception:
                pass
            raise

    @property
    def pending(self):
        return bool(self._pending)

    def _publish(self, serialized):
        from std_msgs.msg import String

        if not self._closed:
            self.publisher.publish(String(data=serialized))

    @staticmethod
    def _identity(snapshot):
        return {
            key: snapshot.get(key)
            for key in ("source_id", "epoch", "request_id", "snapshot_id", "tick", "cloth_tick", "sim_time")
            if type(snapshot.get(key)) in (str, int, float)
            and (not isinstance(snapshot.get(key), float) or math.isfinite(snapshot[key]))
        }

    def _error(self, snapshot, error):
        return json.dumps({**self._identity(snapshot), "error": str(error)}, allow_nan=False)

    def receive(self, msg):
        if self._closed:
            return
        snapshot = {}
        try:
            parsed = json.loads(msg.data)
            if isinstance(parsed, dict):
                snapshot = parsed
            snapshot = validate_snapshot(parsed)
            if self.source_id is not None and snapshot["source_id"] != self.source_id:
                raise ValueError("stale snapshot source_id; restart paired renderer for a new source")
            epoch = snapshot["epoch"]
            if self.epoch is not None and epoch < self.epoch:
                raise ValueError("stale snapshot epoch")
            if self.epoch is None or epoch > self.epoch:
                while self._pending:
                    old = self._pending.popleft()
                    self._publish(self._error(old, "snapshot cancelled by newer epoch"))
                self.source_id, self.epoch = snapshot["source_id"], epoch
                self._payloads.clear()
                self._responses.clear()
            request = snapshot["request_id"]
            canonical = json.dumps(snapshot, sort_keys=True, allow_nan=False)
            if request in self._payloads:
                if self._payloads[request] != canonical:
                    raise ValueError("snapshot request_id payload conflict")
                if request in self._responses:
                    self._publish(self._responses[request])
                return
            self._payloads[request] = canonical
            self._pending.append(snapshot)
        except (ValueError, TypeError) as exc:
            self._publish(self._error(snapshot, exc))

    def process(self):
        from PIL import Image

        if self._closed or not self._pending:
            return False
        snapshot = self._pending.popleft()
        started = time.monotonic()
        try:
            batch = self.capture_backend.capture(lambda: self.apply_snapshot(snapshot))
            images = {}
            for name, rgb in batch["images"].items():
                output = io.BytesIO()
                Image.fromarray(rgb).save(output, format="JPEG", quality=95)
                images[name] = {
                    "encoding": "JPEG",
                    "image_data": base64.b64encode(output.getvalue()).decode(),
                    "height": int(rgb.shape[0]),
                    "width": int(rgb.shape[1]),
                }
            response = {
                **self._identity(snapshot),
                "snapshot": snapshot,
                "images": images,
                "render_frame": batch["render_frame"],
                "render_reference_time": batch["render_reference_time"],
                "swh_frame_number": batch["swh_frame_number"],
                "render_complete_wall_time_ns": time.time_ns(),
                "render_seconds": time.monotonic() - started,
            }
            if self.completed_capture is not None:
                self.completed_capture(response)
            serialized = json.dumps(response, allow_nan=False)
        except Exception as exc:
            serialized = self._error(snapshot, exc)
            self.node.get_logger().error(f"sync capture failed: {exc}")
        self._responses[snapshot["request_id"]] = serialized
        self._publish(serialized)
        return True

    def close(self):
        if self._closed:
            return
        self._closed = True
        errors = []
        for name, destroy in (
            ("subscription", self.node.destroy_subscription),
            ("publisher", self.node.destroy_publisher),
        ):
            resource = getattr(self, name)
            setattr(self, name, None)
            if resource is not None:
                try:
                    destroy(resource)
                except Exception as exc:
                    errors.append(exc)
        backend, self.capture_backend = self.capture_backend, None
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:
                errors.append(exc)
        self._pending.clear()
        if errors:
            raise errors[0]

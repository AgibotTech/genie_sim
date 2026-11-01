# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""ZMQ wire format: a JSON header followed by binary array or image frames.

Result arrays share the SHM transport's shapes and dtypes. Optional JPEG
encoding is recorded in the header. Init/reload sends task_init as JSON;
its response is deferred until the new handler is ready."""

from __future__ import annotations

import contextlib
import json
import sys
from types import SimpleNamespace
from typing import Optional, Union

import numpy as np

# Error code the server may reply with (for step/reset, wire-level "err"
# string) when no handler is active: mid-reload, between the old handler's
# teardown and the new one finishing its build, or after a reload rebuild
# FAILED (ZmqTransport.fail_pending_reload). Defined here (not consumed until
# T2's sim_server main loop) so the constant lives beside the ``reload`` verb
# it complements.
NO_ACTIVE_TASK = "NO_ACTIVE_TASK"

# ResultBuffer fields shipped per response (order = wire order). A reply
# from an older server may still carry "status" (the removed non-blocking
# reset's per-env flag); unpack_result keeps whatever fields arrive, and
# GenieSimVecEnv reports every env READY regardless.
RESULT_FIELDS = [
    "states",
    "joint_states",
    "body_poses",
    "object_poses",
    "images",
    "rewards",
    "terminations",
    "truncations",
    "sim_time",
    "success",
    "global_tick",
    "frame_tick",
]


def _jpeg_codec():
    try:
        import cv2  # noqa: PLC0415

        return ("cv2", cv2)
    except Exception:  # noqa: BLE001
        from PIL import Image  # noqa: PLC0415

        return ("pil", Image)


# One lazily-created JPEG codec pool per process. A step reply handles
# num_envs x cameras images, and doing that serially dominated step time at
# higher env counts. Threads suffice: cv2 and PIL release the GIL in C.
_CODEC_POOL = None


def _codec_pool(n_frames: int):
    """Return the pool when there are enough frames to be worth it, else None.

    Thread count comes from ``RLINF_JPEG_THREADS``, or from the core count
    capped at 16, beyond which scheduling fights cv2's own internal
    parallelism.
    """
    if n_frames < 2:
        return None
    global _CODEC_POOL
    if _CODEC_POOL is None:
        import os  # noqa: PLC0415
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

        try:
            n = int(os.environ.get("RLINF_JPEG_THREADS", "0") or 0)
        except ValueError:
            n = 0
        if n <= 0:
            n = min(16, max(2, (os.cpu_count() or 4) // 2))
        _CODEC_POOL = ThreadPoolExecutor(max_workers=n, thread_name_prefix="rlinf-jpeg")
    return _CODEC_POOL


@contextlib.contextmanager
def _fast_gil_handoff():
    """Reduce GIL handoff latency while worker threads encode or decode frames.

    OpenCV releases the GIL inside codecs; frequent reacquisition can dominate
    small frames in a threaded server. Restore the process-wide switch interval
    on every exit."""
    prev = sys.getswitchinterval()
    sys.setswitchinterval(0.0002)
    try:
        yield
    finally:
        sys.setswitchinterval(prev)


def _map_frames(fn, items: list):
    """Call ``fn`` per frame, in parallel where worthwhile.

    Order must be preserved: ``lens`` is positional, so reordering would send
    env0's image as env3's. ``Executor.map`` preserves order.
    """
    pool = _codec_pool(len(items))
    if pool is None:
        return [fn(x) for x in items]
    with _fast_gil_handoff():
        return list(pool.map(fn, items))


def pack_result(
    out, jpeg_quality: int = 0, include_images: bool = True, image_envs: "Optional[list[int]]" = None
) -> tuple[bytes, list[bytes]]:
    """Serialize result fields into a JSON header and binary frames.

    ``include_images=False`` omits images even if the buffer contains a frame.
    With JPEG encoding, ``image_envs`` selects rows to transfer and the header
    records their indices; omitted rows are zero-filled on decode. Raw image
    transfer (``jpeg_quality=0``) sends the whole buffer and ignores the mask.
    """
    meta: dict = {"fields": []}
    blobs: list[bytes] = []
    for name in RESULT_FIELDS:
        if name == "images" and not include_images:
            continue
        arr = getattr(out, name, None)
        if arr is None:
            continue
        arr = np.asarray(arr)
        if name == "images" and jpeg_quality > 0 and arr.size:
            kind, codec = _jpeg_codec()
            n, c = arr.shape[0], arr.shape[1]
            rows = list(range(n)) if image_envs is None else [int(i) for i in image_envs if 0 <= int(i) < n]

            def _encode(ij, _kind=kind, _codec=codec, _arr=arr, _q=int(jpeg_quality)) -> bytes:
                frame = np.ascontiguousarray(_arr[ij[0], ij[1]])
                if _kind == "cv2":
                    ok, buf = _codec.imencode(
                        ".jpg", frame[..., ::-1], [int(_codec.IMWRITE_JPEG_QUALITY), _q]  # RGB -> BGR for cv2
                    )
                    return buf.tobytes() if ok else b""
                import io  # noqa: PLC0415

                bio = io.BytesIO()
                _codec.fromarray(frame).save(bio, format="JPEG", quality=_q)
                return bio.getvalue()

            parts = _map_frames(_encode, [(i, j) for i in rows for j in range(c)])
            lens = [len(d) for d in parts]
            spec = {
                "name": name,
                "enc": "jpeg",
                "shape": list(arr.shape),
                "dtype": "uint8",
                "lens": lens,
            }
            if image_envs is not None:
                spec["envs"] = rows
            meta["fields"].append(spec)
            blobs.append(b"".join(parts))
        else:
            # NOTE: record the ORIGINAL shape — np.ascontiguousarray promotes
            # 0-d arrays to 1-d (docs: "ndim >= 1"), which silently turned the
            # scalar tick fields () into (1,).
            a = np.ascontiguousarray(arr)
            meta["fields"].append(
                {
                    "name": name,
                    "enc": "raw",
                    "shape": list(arr.shape),
                    "dtype": str(a.dtype),
                }
            )
            blobs.append(a.tobytes())
    # Staged scores: an optional header-only side field. A peer without this
    # key never sees it, and `unpack_result` defaults it to None.
    stage_scores = getattr(out, "stage_scores", None)
    if stage_scores is not None:
        meta["stage_scores"] = stage_scores
    return json.dumps(meta).encode(), blobs


def pack_result_no_images(out) -> tuple[dict, list[bytes]]:
    """Pack all non-image fields from ResultBuffer.

    Returns (meta_dict, blobs) — the meta dict is NOT yet JSON-encoded so
    that it can be merged with the image spec later.  This allows the
    transport to pre-pack scalar/tensor fields while the render barrier is
    still waiting for the GPU, then append image blobs after the copy.
    """
    meta: dict = {"fields": []}
    blobs: list[bytes] = []
    for name in RESULT_FIELDS:
        if name == "images":
            continue
        arr = getattr(out, name, None)
        if arr is None:
            continue
        arr = np.asarray(arr)
        a = np.ascontiguousarray(arr)
        meta["fields"].append(
            {
                "name": name,
                "enc": "raw",
                "shape": list(arr.shape),
                "dtype": str(a.dtype),
            }
        )
        blobs.append(a.tobytes())
    stage_scores = getattr(out, "stage_scores", None)
    if stage_scores is not None:
        meta["stage_scores"] = stage_scores
    return meta, blobs


def pack_images_only(
    out, jpeg_quality: int = 0, image_envs: "Optional[list[int]]" = None
) -> tuple[list[dict], list[bytes]]:
    """Pack only the images field (JPEG encode or raw).

    Returns (field_specs, blobs) — caller appends field_specs to meta and
    blobs to the blob list.
    """
    field_specs: list[dict] = []
    blobs: list[bytes] = []
    arr = getattr(out, "images", None)
    if arr is None:
        return field_specs, blobs
    arr = np.asarray(arr)
    if not arr.size:
        return field_specs, blobs
    if jpeg_quality > 0:
        kind, codec = _jpeg_codec()
        n, c = arr.shape[0], arr.shape[1]
        rows = list(range(n)) if image_envs is None else [int(i) for i in image_envs if 0 <= int(i) < n]

        def _encode(ij, _kind=kind, _codec=codec, _arr=arr, _q=int(jpeg_quality)) -> bytes:
            frame = np.ascontiguousarray(_arr[ij[0], ij[1]])
            if _kind == "cv2":
                ok, buf = _codec.imencode(".jpg", frame[..., ::-1], [int(_codec.IMWRITE_JPEG_QUALITY), _q])
                return buf.tobytes() if ok else b""
            import io  # noqa: PLC0415

            bio = io.BytesIO()
            _codec.fromarray(frame).save(bio, format="JPEG", quality=_q)
            return bio.getvalue()

        parts = _map_frames(_encode, [(i, j) for i in rows for j in range(c)])
        lens = [len(d) for d in parts]
        spec = {
            "name": "images",
            "enc": "jpeg",
            "shape": list(arr.shape),
            "dtype": "uint8",
            "lens": lens,
        }
        if image_envs is not None:
            spec["envs"] = rows
        field_specs.append(spec)
        blobs.append(b"".join(parts))
    else:
        a = np.ascontiguousarray(arr)
        field_specs.append(
            {
                "name": "images",
                "enc": "raw",
                "shape": list(arr.shape),
                "dtype": str(a.dtype),
            }
        )
        blobs.append(a.tobytes())
    return field_specs, blobs


def unpack_result(header: bytes, blobs: list[bytes]) -> SimpleNamespace:
    """(header, frames) -> namespace of numpy arrays (same field contract).

    ``out.image_envs`` holds the indices of the rows the server actually sent
    images for, or ``None`` when every row came (or none did). Rows that were
    not sent are zero-filled.
    """
    meta = json.loads(header.decode())
    out = SimpleNamespace(**{f: None for f in RESULT_FIELDS})
    out.image_envs = None
    for spec, blob in zip(meta["fields"], blobs):
        name, shape = spec["name"], tuple(spec["shape"])
        if spec["enc"] == "jpeg":
            kind, codec = _jpeg_codec()
            n, c = shape[0], shape[1]
            rows = spec.get("envs")
            # Rows that were not sent must be zero, not uninitialised memory:
            # np.empty garbage looks like a noise image and hides the fact
            # that the row was never requested.
            arr = np.empty(shape, dtype=np.uint8) if rows is None else np.zeros(shape, dtype=np.uint8)
            if rows is None:
                rows = list(range(n))
            # Slice per-frame byte ranges first, then decode on the pool.
            tasks = []
            off = 0
            for k, ln in enumerate(spec["lens"][: len(rows) * c]):
                tasks.append((rows[k // c], k % c, blob[off : off + ln]))
                off += ln

            def _decode(t, _kind=kind, _codec=codec, _arr=arr) -> None:
                i, j, data = t
                if not data:
                    _arr[i, j] = 0
                    return
                if _kind == "cv2":
                    img = _codec.imdecode(np.frombuffer(data, np.uint8), _codec.IMREAD_COLOR)
                    _arr[i, j] = img[..., ::-1]  # BGR -> RGB
                else:
                    import io  # noqa: PLC0415

                    _arr[i, j] = np.asarray(_codec.open(io.BytesIO(data)).convert("RGB"))

            _map_frames(_decode, tasks)
            setattr(out, name, arr)
            if spec.get("envs") is not None:
                # Which rows are real, so consumers can test rather than
                # guess from whether an image looks black.
                out.image_envs = list(spec["envs"])
        else:
            setattr(out, name, np.frombuffer(blob, dtype=np.dtype(spec["dtype"])).reshape(shape).copy())
    # Absent on old-header packets (no "stage_scores" key) -> None.
    out.stage_scores = meta.get("stage_scores")
    return out


def pack_request(
    verb: str,
    payload: Optional[Union[np.ndarray, dict]] = None,
    need_images: bool = True,
) -> list[bytes]:
    """Serialize a request with a numeric payload or an init/reset mapping.

    need_images may be a boolean or a per-environment mask. Omit the header
    flag when all images are requested so older peers retain the default.
    A legacy reset block flag is ignored by servers with synchronous reset."""
    extra = {}
    mask = _as_image_mask(need_images)
    if mask is not None:
        # An all-True mask degenerates to writing no key at all and an
        # all-False one to need_images: false, so the new key appears only
        # when the request really is partial.
        if all(mask):
            pass
        elif not any(mask):
            extra["need_images"] = False
        else:
            extra["need_images"] = True
            extra["need_images_mask"] = [bool(m) for m in mask]
    elif not need_images:
        extra["need_images"] = False
    if isinstance(payload, dict):
        head = json.dumps(
            {
                "verb": verb,
                "shape": None,
                "dtype": None,
                "task_init": payload,
                **extra,
            }
        ).encode()
        return [head]
    head = json.dumps(
        {
            "verb": verb,
            "shape": list(payload.shape) if payload is not None else None,
            "dtype": str(payload.dtype) if payload is not None else None,
            **extra,
        }
    ).encode()
    frames = [head]
    if payload is not None:
        frames.append(np.ascontiguousarray(payload).tobytes())
    return frames


def unpack_request(
    frames: list[bytes],
) -> tuple[str, Optional[Union[np.ndarray, dict]]]:
    head = json.loads(frames[0].decode())
    task_init = head.get("task_init")
    if task_init is not None:
        return head["verb"], task_init
    payload = None
    if head.get("shape") is not None and len(frames) > 1:
        payload = np.frombuffer(frames[1], dtype=np.dtype(head["dtype"])).reshape(head["shape"]).copy()
    return head["verb"], payload


def _as_image_mask(need_images) -> "Optional[list[bool]]":
    """Return a list[bool] when ``need_images`` is a per-env mask, else None.

    ``True``, ``np.True_``, and ``[True]`` look alike at the call site:
    ``np.bool_`` is a scalar and not a mask, while a length-1 list is one.
    Getting it wrong silently turns "only env3's image" into all or none.
    """
    if need_images is None or isinstance(need_images, (bool, np.bool_)):
        return None
    if isinstance(need_images, np.ndarray):
        return [bool(x) for x in need_images.reshape(-1)]
    if isinstance(need_images, (list, tuple)):
        return [bool(x) for x in need_images]
    return None


def request_needs_images(frames: list[bytes]) -> bool:
    """Read the image-request flag, defaulting to ``True`` if absent."""
    head = json.loads(frames[0].decode())
    return bool(head.get("need_images", True))


def request_image_envs(frames: list[bytes]) -> "Optional[list[int]]":
    """Env indices of the request's per-env image mask, or ``None`` when
    there is no mask (all rows or none, per ``request_needs_images``).

    Separate from ``request_needs_images`` because the call site destructures
    ``unpack_request`` into two values and cannot take a third.
    """
    head = json.loads(frames[0].decode())
    mask = head.get("need_images_mask")
    if not mask:
        return None
    return [i for i, m in enumerate(mask) if m]


__all__ = [
    "RESULT_FIELDS",
    "NO_ACTIVE_TASK",
    "pack_result",
    "unpack_result",
    "pack_request",
    "unpack_request",
    "request_needs_images",
    "request_image_envs",
]

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Shared-memory transport for de-tiled RGB frames.

The render node creates the segment and publishes an int64 header followed by
a contiguous uint8 image buffer. Frame ids increase across stage reloads;
consumers also check generation and tick metadata before accepting a frame."""

from __future__ import annotations

from multiprocessing.shared_memory import SharedMemory

import numpy as np

# Header: int64[8] = [magic, num_envs, height, width, channels, frame_id,
#                     rendered_tick, generation].
_HEADER_LEN = 8
_HEADER_BYTES = _HEADER_LEN * 8
_MAGIC = 0x54494C4446524D31  # "TILDFRM1"


class TiledFrameShm:
    """One de-tiled RGB frame in shared memory: a header + an ``(N,H,W,3)``
    uint8 image. Writer (render node) uses ``create=True`` then ``publish``;
    reader (bridge) uses ``create=False`` then polls ``frame_id`` / ``read``."""

    def __init__(
        self, name: str, num_envs: int, height: int, width: int, channels: int = 3, *, create: bool = False
    ) -> None:
        self.num_envs = int(num_envs)
        self.height = int(height)
        self.width = int(width)
        self.channels = int(channels)
        self._img_bytes = self.num_envs * self.height * self.width * self.channels
        size = _HEADER_BYTES + self._img_bytes
        if create:
            self._shm = SharedMemory(name=name, create=True, size=size)
        else:
            self._shm = SharedMemory(name=name, create=False)
            if self._shm.size < size:
                self._shm.close()
                raise ValueError(f"tiled shm '{name}' too small: {self._shm.size} < {size}")
        self._hdr = np.ndarray((_HEADER_LEN,), dtype=np.int64, buffer=self._shm.buf[:_HEADER_BYTES])
        self.rgb = np.ndarray(
            (self.num_envs, self.height, self.width, self.channels),
            dtype=np.uint8,
            buffer=self._shm.buf[_HEADER_BYTES : _HEADER_BYTES + self._img_bytes],
        )
        if create:
            self._hdr[:] = [_MAGIC, self.num_envs, self.height, self.width, self.channels, 0, 0, 0]
        else:
            self._validate_header()

    def _validate_header(self) -> None:
        magic, n, h, w, c, _fid, _tick, _gen = (int(x) for x in self._hdr)
        if magic != _MAGIC:
            raise ValueError(f"tiled shm bad magic {magic:#x} (want {_MAGIC:#x})")
        if (n, h, w, c) != (self.num_envs, self.height, self.width, self.channels):
            raise ValueError(
                f"tiled shm dims {(n, h, w, c)} != expected "
                f"{(self.num_envs, self.height, self.width, self.channels)}"
            )

    @property
    def name(self) -> str:
        return self._shm.name

    @property
    def frame_id(self) -> int:
        return int(self._hdr[5])

    @property
    def rendered_tick(self) -> int:
        return int(self._hdr[6])

    @property
    def generation(self) -> int:
        return int(self._hdr[7])

    def publish(self, frame_id: int, rendered_tick: int = 0) -> None:
        """Set rendered_tick THEN frame_id — readers key on frame_id, so tick
        must be visible before the frame_id bump they poll on."""
        self._hdr[6] = int(rendered_tick)
        self._hdr[5] = int(frame_id)

    def reset_for_reload(self, generation: int) -> None:
        """Stage hot-swap: bump ``generation`` and zero ``rendered_tick`` in
        place — the segment (and ``frame_id``) is REUSED as-is, never
        unlinked/recreated (dims are unchanged, guaranteed by the caller's
        applicability gate). Order matches ``publish``'s reader contract:
        tick before any field a reader might key a barrier decision on."""
        self._hdr[6] = 0
        self._hdr[7] = int(generation)

    def close(self, *, unlink: bool = False) -> None:
        # drop numpy views into the buffer before closing (else BufferError)
        self._hdr = None
        self.rgb = None
        try:
            self._shm.close()
        finally:
            if unlink:
                try:
                    self._shm.unlink()
                except FileNotFoundError:
                    pass


__all__ = ["TiledFrameShm"]

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pinned-host ping-pong mirror of a Newton GPU buffer.

Why this exists
---------------
Reading ``state_0.joint_q`` (or any device-side ``wp.array``) from
the publish thread via ``.numpy()`` issues a SYNCHRONOUS device→host
transfer that blocks until *all queued GPU work on that stream* has
completed.  At 100 Hz on a G2-class scene that's ~6 ms per tick of
pure CPU stall, multiplied across joints + bodies + odom — enough to
push the publish phase over the 10 ms budget all by itself.

The fix: keep a pinned-host buffer that the engine refreshes via
``wp.copy(host_pinned, src, stream=…)`` enqueued onto the same stream
the captured physics graph runs on, then have the publish path read
that buffer directly with no further GPU contact.  Pinned memory
+ stream-ordered memcpy means the copy is asynchronous: the host
returns immediately and the DMA engine finishes the move while the
next tick's graph runs.

Why a single buffer is unsafe
-----------------------------
A single host buffer plus a single ``wp.Event`` object DOES NOT WORK,
even when "rotating" the references each tick:

    self._event_prev = self._event   # both point to E
    stream.record_event(self._event) # mutates E in place

``wp.Event`` is a CUDA event handle.  ``record_event`` overwrites the
event's recorded position in place — there is no immutable snapshot.
So ``self._event_prev`` and ``self._event`` end up pointing at the
*same* recording, and ``synchronize_event(self._event_prev)`` waits
for the LATEST tick's graph + copy to finish, defeating the whole
point.  The pinned-host buffer is also race-prone: while
``get_joint_states`` reads it on the host, ``step`` could be enqueuing
a fresh ``wp.copy`` into the same memory.

Ping-pong design
----------------
Two host buffers, alternated by a writer index:

    refresh(src):
        slot = self._next_writer
        wp.copy(buf[slot], src, stream)
        self._next_writer = 1 - slot       # flip

    read():
        # Return the OTHER slot — the one written *last* tick, not
        # the one just written this tick.  ``self._next_writer``
        # points at the slot that will be overwritten next; it
        # currently still holds the previous-tick data.
        return views[next_writer]

Tick 1 has nothing on the "other" slot yet (it was never written),
so ``read`` returns ``None`` and the caller takes the synchronous
fallback ONCE.  From tick 2 onward ``read`` returns the slot written by
the previous refresh, once that slot's event says its copy has landed
(see "Event query, not wall-time inference" below).

Each side now owns immutable state for the duration it's being
read: refresh writes slot A while read reads slot B.

Event query, not wall-time inference
------------------------------------
``read`` guards the slot with a per-slot ``wp.Event``: ``refresh``
records one right after its ``wp.copy``, and ``read`` checks
``event.is_complete`` before handing the buffer out.  The check is a
NON-BLOCKING status query (microseconds), not ``wp.synchronize_event``
(which on warp 1.14 burned 2-6 ms per tick — and did so pointlessly,
because the single-event bug above meant it waited on the wrong
recording).  Not complete yet -> return ``None`` -> the caller takes
its existing synchronous ``src.numpy()`` fallback.

Do not infer completion purely from **wall-time separation** ("the slot
was filled ≥ one physics_dt ago and the DMA is microseconds, so it must
have landed"): that inference holds only while the CPU stays behind the
GPU, and it does not always.

The host enqueues work asynchronously and returns immediately, so a
fast client can let the CPU run ahead of the GPU: when the caller only
waits on the render barrier once per action chunk instead of once per
substep, commands arrive back-to-back and several ticks' worth of graph
launches + copies can queue up before the publish thread calls
``read``.  The GPU may not have executed that slot's copy yet, so the
buffer still holds what the PREVIOUS write to that slot left there —
two ticks stale, not one.

A read artifact from this failure mode can look exactly like a
mode-dependent trajectory divergence, with one tell that distinguishes
it from real physics divergence: it re-converges exactly at every chunk
boundary — the one step where both paths wait on the render barrier
and the GPU catches up. Real divergence does not heal on a period. The
event query avoids the failure mode entirely by checking the slot's
actual state instead of assuming it from elapsed time.

Cost: an ``is_complete`` query on the happy path (microseconds) and the
synchronous fallback only on the ticks that actually lose the race.

Trade-off
---------
The mirror lags by ONE physics tick — the latest captured-graph
output is in the buffer the writer is currently filling, not the one
the reader sees.  At 100 Hz that's 10 ms of /joint_states lag, well
inside what RViz / WBC / state observation already tolerate from
ROS DDS jitter.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import warp as wp


class AsyncMirror:
    """Two-slot pinned-host mirror of a single Newton GPU array.

    Owns:
      * Two ``wp.array`` buffers on ``device="cpu", pinned=True``
      * A writer index that ping-pongs after each ``refresh``
      * One ``wp.Event`` per slot, recorded right after that slot's copy

    See the module docstring for why each slot needs its own event.

    Lifecycle:
      * Construct with no arguments at engine init.
      * Call ``refresh(src, stream)`` once per physics step.  First
        call sizes the buffers from ``src`` and records the first
        event; subsequent calls reuse the buffers in place.
      * Call ``read()`` from the publish thread.  Returns ``None``
        until the first ``refresh`` has flipped the writer (i.e. the
        publish thread runs after step has called refresh at least
        once).  Callers should fall back to the synchronous
        ``src.numpy()`` path during this brief warmup window.
    """

    __slots__ = (
        "_bufs",
        "_views",
        "_events",
        "_next_writer",
        "_refresh_count",
        "_failed",
    )

    def __init__(self) -> None:
        self._bufs: list = [None, None]
        self._views: list = [None, None]
        # One **independent** event per slot -- the same object must not be
        # reused. ``wp.Event`` is a CUDA event handle; ``record_event``
        # overwrites its recorded position in place, with no immutable
        # snapshot. Reusing one object would make "last tick's event" and
        # "this tick's event" point at the same recording (this is exactly
        # the pitfall the module-level comment above describes). Only two
        # slots with two separate objects can each independently answer
        # "has this slot's copy landed?".
        self._events: list = [None, None]
        # Index of the slot the next ``refresh`` will write into.
        # After a refresh, this flips to the OTHER slot — so ``read``
        # uses ``self._next_writer`` itself as the index of the slot
        # to read from (the one the previous refresh just finished
        # filling, which the next refresh will overwrite).
        self._next_writer: int = 0
        # Total successful refreshes so far.  ``read`` returns None
        # while this is < 2 — there's no completed "other" slot yet.
        self._refresh_count: int = 0
        # Latched on any allocation / copy failure.  Stops further
        # work permanently and forces ``read`` to return None so the
        # caller takes the synchronous fallback.
        self._failed: bool = False

    @property
    def ok(self) -> bool:
        """True once both slots have been written at least once.

        After 2 successful refreshes, slot 0 holds tick 1's data and
        slot 1 holds tick 2's data; from tick 3 onward, ``read`` can
        always return the previous tick's data from the slot the
        next refresh is about to overwrite.
        """
        return not self._failed and self._refresh_count >= 2

    def refresh(self, src: Any, stream: Any) -> None:
        """Enqueue an async device→host copy on ``stream``.

        ``src`` is the device-side ``wp.array`` to mirror (typically
        ``state_0.joint_q`` etc.).  ``stream`` is the warp stream the
        captured physics graph just ran on — the copy must land on the
        same stream so it's FIFO-ordered behind the kernel writes.

        First call allocates both pinned-host buffers sized to
        ``src``.  Subsequent calls reuse them in place; if ``src``
        ever changes shape (defensive — Newton's state buffers don't
        change size at runtime), the slot is reallocated.

        Records the slot's event right after the copy, so ``read``
        can check whether this copy has actually landed instead of
        assuming it from elapsed time.
        """
        if self._failed:
            return
        try:
            slot = self._next_writer
            buf = self._bufs[slot]
            if buf is None or buf.size != src.size or buf.dtype != src.dtype:
                buf = wp.empty(src.shape, dtype=src.dtype, device="cpu", pinned=True)
                self._bufs[slot] = buf
                self._views[slot] = buf.numpy()
            wp.copy(buf, src, stream=stream)
            # Record the event right after the copy: it becomes the proof
            # that "this slot's copy has completed". FIFO ordering on the
            # same stream means event-complete == copy-complete.
            ev = self._events[slot]
            if ev is None:
                ev = wp.Event()
                self._events[slot] = ev
            stream.record_event(ev)
            self._next_writer = 1 - slot
            self._refresh_count += 1
        except Exception:
            self._failed = True

    def synchronize(self) -> None:
        """Wait for queued copies before releasing this mirror's pinned buffers.

        Only used when retiring a consumer, never on the normal step/read path.
        A failed refresh may have queued a copy without recording its event.
        """
        if self._failed:
            wp.synchronize()
        else:
            for event in self._events:
                if event is not None:
                    wp.synchronize_event(event)

    def read(self) -> Optional[np.ndarray]:
        """Return the numpy view of "the slot the next refresh will
        overwrite".

        That slot holds last tick's data (the most recent refresh wrote the
        OTHER slot and flipped this index).

        Guarded by a per-slot ``wp.Event`` rather than wall-clock inference --
        see the module docstring's "Event query, not wall-time inference"
        section for why wall-clock inference is unsafe. The ``is_complete``
        check is a non-blocking status query (microsecond-scale), not the
        blocking wait of ``wp.synchronize_event``. If the query finds the
        copy hasn't landed, returns ``None`` and lets the caller fall back to
        the existing synchronous ``src.numpy()`` path.

        Still returns ``None`` during warmup (refresh_count < 2) or once
        failed.
        """
        if not self.ok:
            return None
        slot = self._next_writer
        ev = self._events[slot]
        if ev is not None:
            try:
                if not ev.is_complete:
                    # The GPU hasn't executed this copy yet -- the slot
                    # holds stale data and can't be used.
                    return None
            except Exception:
                # The query itself failed; fall back to the sync path
                # rather than guessing.
                return None
        return self._views[slot]

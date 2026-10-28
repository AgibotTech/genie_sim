# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Benchmark-side data publishers (eval checker support).

Publishers for data the *benchmark / eval* harness consumes — not for
human-facing debug visualization.  That distinction is why these live
here and not in ``visualizers/markers.py``: markers are RViz debug aids
that can be turned off freely, while everything in this module is part of
a wire contract an external consumer depends on.

Current publishers:
  * ``publish_body_aabb_local`` — one-shot latched per-body local AABB
    table on ``/rlinf/body_aabb_local``, read by rlinf's eval checkers.
    Called once at engine startup from ``setup/init_pose.py``.

The publisher outlives the call, deliberately
---------------------------------------------
This module used to create an *ephemeral* node, publish, and destroy it on
the next line, on the stated reasoning that "TRANSIENT_LOCAL keeps the
sample alive for late subscribers even after the node that published it is
gone".  That is not what TRANSIENT_LOCAL does.  ROS 2 durability is
**writer-side**: the policy makes the *publisher* responsible for replaying
its history to late-joining subscriptions, so the latch dies with the
writer.  (DDS ``TRANSIENT`` / ``PERSISTENT`` durability would outlive it —
rmw exposes neither.)

Measured on Humble + Fast DDS with matching QoS on both ends, a subscriber
attaching 0.5 s after the publish received nothing when the writer had been
destroyed and received the sample when it had not.  The one consumer is
rlinf's wrapper, in another process, attaching after engine build — so the
channel listed in ``ros2 topic list``, errored nowhere, and had never
carried data.

The node is therefore kept in a module global for the life of the process
and torn down from ``PhysicsEngine.shutdown``.  Nothing spins it: durability
replay happens in the middleware, not in an rclpy executor, which is why a
publisher with no executor still serves late joiners.
"""

from __future__ import annotations

from typing import Any, Optional

#: Retained writer.  Its LIFETIME is the latch — see the module docstring.
_node: Any = None
_publisher: Any = None
_static_publisher: Any = None
_rclpy: Any = None


def publish_body_aabb_local(
    names: Any,
    aabb_local: Any,
    topic: str = "/rlinf/body_aabb_local",
    logger: Any = None,
) -> None:
    """One-shot LATCHED publish of the engine's static per-body LOCAL AABB
    table, consumed by rlinf's wrapper-side ``get_body_aabbs`` (see
    ``rlinf_geniesim``'s ``physics/backends/_ros_bridge.py`` /
    ``_upstream_base.py``).

    Fires EXACTLY ONCE at engine startup, unconditionally, whenever the
    active solver provides ``body_aabb_local``.  The publisher it creates is
    retained until :func:`shutdown`, because destroying it un-latches the
    sample (module docstring).

    Wire contract (do not change without updating the rlinf wrapper too):
      * topic   ``/rlinf/body_aabb_local`` — ABSOLUTE (leading ``/`` makes
        this land at the exact path regardless of this node's
        ``/genie_sim_engine`` namespace).
      * type    ``std_msgs/String``.
      * QoS     RELIABLE + TRANSIENT_LOCAL, depth=1 (latched — a
        subscriber that joins after this one publish still receives it,
        for as long as this process lives).
      * payload JSON ``{"names": [...], "aabb_local": [[cx,cy,cz,hx,hy,hz],
        ...]}``, ``names[i]`` aligned with ``aabb_local[i]``.

    A subscriber must declare TRANSIENT_LOCAL too.  A VOLATILE reader does
    not match a TRANSIENT_LOCAL writer's offer, and DDS declines the
    connection silently — the topic still lists, the data never arrives.

    Args:
        names: per-env bare body labels (NO leading ``/``, NO
            ``env_<w>/`` prefix) — must be the SAME strings (same order)
            the wrapper resolves ``_body_names`` / ``body_link_names`` to,
            i.e. the caller should pass the SAME per-env name list used to
            build the ``/tf_render`` frames (see ``common/session.py``).
        aabb_local: array-like ``(len(names), 6)`` — ``[cx,cy,cz,hx,hy,hz]``
            (body-frame center + half-extent) per body, aligned with
            ``names``.
        topic: override topic name (default ``/rlinf/body_aabb_local``).
        logger: optional logger for info/warn messages.

    Never raises — logs a warning (if ``logger`` is given) and no-ops on
    any failure. This is a nice-to-have for eval checkers, never worth
    crashing engine startup over.
    """
    global _node, _publisher, _rclpy
    if not names:
        return
    try:
        import json  # noqa: PLC0415

        import numpy as _np  # noqa: PLC0415
        import rclpy  # noqa: PLC0415
        from rclpy.qos import (  # noqa: PLC0415
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from std_msgs.msg import String  # noqa: PLC0415

        if not rclpy.ok():
            rclpy.init()
        _rclpy = rclpy

        table = _np.asarray(aabb_local, dtype=_np.float32).reshape(len(names), 6)
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        if _node is None:
            _node = rclpy.create_node("benchmark_publisher", namespace="genie_sim_engine")
        if _publisher is None:
            _publisher = _node.create_publisher(String, topic, qos)
        msg = String()
        msg.data = json.dumps({"names": list(names), "aabb_local": table.tolist()})
        _publisher.publish(msg)

        if logger is not None:
            logger.info(
                f"[newton-standalone] body_aabb_local: published "
                f"{len(names)} body local AABB(s) on {topic} (latched; "
                f"writer held until engine shutdown)"
            )

    except Exception as exc:  # noqa: BLE001
        if logger is not None:
            logger.warn(f"[newton-standalone] body_aabb_local publish failed: {exc!r}")


def publish_static_prop_aabb(names, boxes, logger=None) -> None:
    """Publish the static props' world AABBs on ``/rlinf/static_prop_aabb``, latched.

    Only a scene that names ``newton.eval_static_props`` has anything to send,
    so an empty table before any publish creates no node and no topic. Once a
    writer exists, an empty table is still sent: after a reload it clears the
    previous scene's boxes. The writer is retained until ``shutdown``. A failure
    is logged, never raised, so it cannot stop the engine from coming up.
    """
    global _node, _static_publisher, _rclpy
    if not names and _static_publisher is None:
        return
    try:
        import json  # noqa: PLC0415

        import rclpy  # noqa: PLC0415
        from rclpy.qos import (  # noqa: PLC0415
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from std_msgs.msg import String  # noqa: PLC0415

        if not rclpy.ok():
            rclpy.init()
        _rclpy = rclpy
        if _node is None:
            _node = rclpy.create_node("benchmark_publisher", namespace="genie_sim_engine")
        if _static_publisher is None:
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            _static_publisher = _node.create_publisher(String, "/rlinf/static_prop_aabb", qos)
        msg = String()
        msg.data = json.dumps({"names": list(names), "aabb_world": boxes})
        _static_publisher.publish(msg)
        if logger is not None:
            logger.info(f"[newton] published {len(names)} static world AABBs (latched)")
    except Exception as exc:  # noqa: BLE001
        if logger is not None:
            logger.warn(f"[newton] static prop AABB publish failed: {exc!r}")


def shutdown() -> None:
    """Best-effort teardown — called from the engine on shutdown.

    Safe to call multiple times, and safe to call when nothing was ever
    published.  Leaves the rclpy context alone: other parts of the process
    (markers, the wrapper's own node) may still rely on it.

    Tearing this down ends the latch, which is correct — the table describes
    a model that no longer exists once the engine is gone.
    """
    global _node, _publisher, _static_publisher
    _publisher = None
    _static_publisher = None
    if _node is not None:
        try:
            _node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        _node = None


def _retained() -> Optional[Any]:
    """The retained node, for tests that pin the lifetime contract."""
    return _node

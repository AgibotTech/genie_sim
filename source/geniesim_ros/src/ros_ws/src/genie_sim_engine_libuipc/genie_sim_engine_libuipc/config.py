# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Validated scene-YAML configuration for the Newton SolverIPC adapter."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class IPCConfig:
    """SolverIPC tuning and robot collision selection parsed from a scene."""

    contact_d_hat: float = 0.002
    cloth_thickness: float = 0.001
    abd_kappa: float = 1.0e8
    kinematic_constraint_strength: tuple[float, float] = (1000.0, 1000.0)
    collision_link_names: tuple[str, ...] = ()
    gripper_self_contact: bool = False
    robot_static_collision: bool = True
    linear_system_solver: str = "fused_pcg"
    abd_full_block_preconditioner: bool = False
    linear_system_graph_enable: bool = True
    linear_system_fused_preconditioner_enable: bool = True
    newton_max_iter: int | None = None
    friction_rate: float | None = None
    linear_system_tol_rate: float | None = None
    linear_system_check_interval: int | None = None
    newton_semi_implicit: bool | None = None
    newton_semi_implicit_beta_tol: float | None = None
    fem_mas_partition_size: int | None = None
    cloth_bending_yield_stress: float | None = None
    cloth_bending_hardening_modulus: float | None = None

    @classmethod
    def from_scene(cls, scene_cfg: dict[str, Any] | None) -> "IPCConfig":
        """Parse and validate ``libuipc`` from a scene configuration."""
        raw = (scene_cfg or {}).get("libuipc") or {}
        for key, default in (("contact_d_hat", 0.002), ("cloth_thickness", 0.001), ("abd_kappa", 1.0e8)):
            value = raw.get(key, default)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"libuipc.{key} must be finite and positive")
        strength = tuple(float(value) for value in raw.get("kinematic_constraint_strength", (1000.0, 1000.0)))
        if len(strength) != 2:
            raise ValueError("libuipc.kinematic_constraint_strength must contain exactly two values")
        names = raw.get("collision_link_names", raw.get("gripper_link_names", ())) or ()
        if isinstance(names, str):
            raise ValueError("libuipc.collision_link_names must be a sequence of link names")
        robot_static_collision = raw.get("robot_static_collision", True)
        if not isinstance(robot_static_collision, bool):
            raise ValueError("libuipc.robot_static_collision must be a boolean")
        if not robot_static_collision and not names:
            raise ValueError("robot_static_collision: false requires explicit collision_link_names")
        max_iter = raw.get("newton_max_iter")
        if max_iter is not None and (isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1):
            raise ValueError("libuipc.newton_max_iter must be a positive integer or null")
        optional = {}
        partition = raw.get("fem_mas_partition_size")
        if partition is not None:
            if isinstance(partition, bool) or not isinstance(partition, int) or not 1 <= partition <= 16:
                raise ValueError("libuipc.fem_mas_partition_size must be an integer in [1, 16]")
            optional["fem_mas_partition_size"] = partition
        for key in ("cloth_bending_yield_stress", "cloth_bending_hardening_modulus"):
            value = raw.get(key)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"libuipc.{key} must be finite")
            if value < 0 or (key == "cloth_bending_yield_stress" and value == 0):
                raise ValueError(f"libuipc.{key} is outside its physical range")
            optional[key] = float(value)
        if optional.get("cloth_bending_hardening_modulus", 0) and "cloth_bending_yield_stress" not in optional:
            raise ValueError("cloth_bending_hardening_modulus requires cloth_bending_yield_stress")
        for key in ("friction_rate", "linear_system_tol_rate", "newton_semi_implicit_beta_tol"):
            value = raw.get(key)
            if value is None:
                continue
            value = float(value)
            if not math.isfinite(value) or value < 0 or (key != "friction_rate" and value == 0):
                raise ValueError(
                    f"libuipc.{key} must be finite and {'nonnegative' if key == 'friction_rate' else 'positive'}"
                )
            if key == "newton_semi_implicit_beta_tol" and value > 1:
                raise ValueError("libuipc.newton_semi_implicit_beta_tol must be in (0, 1]")
            optional[key] = value
        interval = raw.get("linear_system_check_interval")
        if interval is not None:
            if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
                raise ValueError("libuipc.linear_system_check_interval must be a positive integer")
            optional["linear_system_check_interval"] = interval
        semi_implicit = raw.get("newton_semi_implicit")
        if semi_implicit is not None:
            if not isinstance(semi_implicit, bool):
                raise ValueError("libuipc.newton_semi_implicit must be a boolean")
            optional["newton_semi_implicit"] = semi_implicit
        return cls(
            contact_d_hat=float(raw.get("contact_d_hat", 0.002)),
            cloth_thickness=float(raw.get("cloth_thickness", 0.001)),
            abd_kappa=float(raw.get("abd_kappa", 1.0e8)),
            kinematic_constraint_strength=(strength[0], strength[1]),
            collision_link_names=tuple(str(name) for name in names),
            gripper_self_contact=bool(raw.get("gripper_self_contact", False)),
            robot_static_collision=robot_static_collision,
            linear_system_solver=str(raw.get("linear_system_solver", "fused_pcg")),
            abd_full_block_preconditioner=bool(raw.get("abd_full_block_preconditioner", False)),
            linear_system_graph_enable=bool(raw.get("linear_system_graph_enable", True)),
            linear_system_fused_preconditioner_enable=bool(raw.get("linear_system_fused_preconditioner_enable", True)),
            newton_max_iter=None if max_iter is None else int(max_iter),
            **optional,
        )

    def solver_options(self, *, urdf_body_prefix: str, length_scale: float = 1.0) -> dict[str, Any]:
        """Return keyword arguments accepted by :class:`SolverIPC`."""
        if not math.isfinite(length_scale) or length_scale <= 0:
            raise ValueError("IPC length_scale must be finite and positive")
        options: dict[str, Any] = {
            "contact_d_hat": self.contact_d_hat * length_scale,
            "cloth_thickness": self.cloth_thickness * length_scale,
            "abd_kappa": self.abd_kappa,
            "kinematic_constraint_strength": self.kinematic_constraint_strength,
            "gripper_link_names": self.collision_link_names or None,
            "gripper_self_contact": self.gripper_self_contact,
            "urdf_body_prefix": urdf_body_prefix,
            "linear_system_solver": self.linear_system_solver,
            "abd_full_block_preconditioner": self.abd_full_block_preconditioner,
            "linear_system_graph_enable": self.linear_system_graph_enable,
            "linear_system_fused_preconditioner_enable": self.linear_system_fused_preconditioner_enable,
        }
        if self.newton_max_iter is not None:
            options["newton_max_iter"] = self.newton_max_iter
        for key in (
            "friction_rate",
            "linear_system_tol_rate",
            "linear_system_check_interval",
            "newton_semi_implicit",
            "newton_semi_implicit_beta_tol",
            "fem_mas_partition_size",
            "cloth_bending_yield_stress",
            "cloth_bending_hardening_modulus",
        ):
            value = getattr(self, key)
            if value is not None:
                options[key] = value
        return options


__all__ = ["IPCConfig"]

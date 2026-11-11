# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Recording fallback centres follow the aligned wrist's local z axis."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


@pytest.mark.parametrize("gripper,offset", [("omnipicker", 0.14845), ("ctek90d", 0.185)])
def test_center_offset_preserves_wrist_orientation(gripper, offset):
    path = Path(__file__).resolve().parents[1] / "src/geniesim_teleop/data_recording/derive_fields.py"
    spec = importlib.util.spec_from_file_location("recording_frames", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rotation = Rotation.from_euler("xyz", [0.3, -0.4, 0.7])
    wrist = np.array([0.2, -0.1, 1.4])
    center, quaternion = module.effector_from_arm_end(wrist, rotation.as_quat(), gripper)
    np.testing.assert_allclose(rotation.inv().apply(center - wrist), [0, 0, offset], atol=1e-12)
    assert (Rotation.from_quat(quaternion) * rotation.inv()).magnitude() < 1e-12

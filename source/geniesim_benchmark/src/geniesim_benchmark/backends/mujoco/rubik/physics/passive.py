# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Conservative internal cubic detents; independent of goals, moves and grasp state.

Potential U=-k/4 sum(R_ij**4), with 24 equivalent axis-aligned minima. Equal and
opposite torques act on a cubelet and its core. No translational force, target
angle, state snapping or motor is used. This approximates a magnetic cube's
passive alignment, rather than reproducing its internal CAD mechanism.
"""
import numpy as np
import mujoco
from .cube import coords

def guide_torque(R,stiffness):
    """Gradient of k/2 (1-max(R_ij²)), a passive internal axis guide.

    Every rotation about any cardinal core axis is unconstrained. Only tilt
    away from that family is resisted. The closest guide is chosen from the
    current orientation, with no move, layer, target angle or controller input.
    """
    flat=np.abs(R).reshape(-1,9).argmax(axis=1);rows,columns=flat//3,flat%3
    batch=np.arange(len(R));vectors=R[batch,:,columns];axes=np.eye(3)[rows]
    return stiffness*R[batch,rows,columns,None]*np.cross(vectors,axes)


def install_detents(m,stiffness=.02,center_stiffness=None,guide_stiffness=0.):
    root=m.body('cube:middle').id
    cubies=[b for b in range(m.nbody) if (m.body(b).name or '').startswith('cube:cubelet:')]
    # Axle and cubie detents are fixed before stepping and are independent of
    # the controller. Both use the same fourfold potential without a goal face.
    center_stiffness=stiffness if center_stiffness is None else center_stiffness
    strengths=np.array([center_stiffness if np.count_nonzero(coords(m.body(b).name))==1 else stiffness for b in cubies])
    zero=np.zeros(3)
    def passive(model,d):
        R0=d.xmat[root].reshape(3,3)
        R=R0.T@d.xmat[cubies].reshape(-1,3,3)
        local=strengths[:,None]*np.cross(R.transpose(0,2,1),(R**3).transpose(0,2,1)).sum(axis=1)
        if guide_stiffness:local+=guide_torque(R,guide_stiffness)
        torques=local@R0.T
        for b,torque in zip(cubies,torques):
            mujoco.mj_applyFT(model,d,zero,torque,d.xpos[root],b,d.qfrc_passive)
        mujoco.mj_applyFT(model,d,zero,-torques.sum(axis=0),d.xpos[root],root,d.qfrc_passive)
    mujoco.set_mjcb_passive(passive)
    return passive

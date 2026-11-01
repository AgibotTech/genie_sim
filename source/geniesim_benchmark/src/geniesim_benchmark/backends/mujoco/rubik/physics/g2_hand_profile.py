# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Human-like planning bounds for the native Wuji joints; no model edits."""
import mujoco
import numpy as np
from scipy.optimize import least_squares

from .g2_closed_loop import FINGERS, G2Controller


def mirror_hand_pose(hand):
    """Mirror native Wuji angles; ordinary abduction reverses, flexion does not."""
    mirrored=np.asarray(hand,dtype=float).copy()
    if mirrored.shape!=(20,) or not np.isfinite(mirrored).all():
        raise ValueError('Expected twenty finite native hand joint angles')
    mirrored[[5,9,13,17]]*=-1
    return mirrored


class NaturalHandProfile:
    # The four ordinary fingers have flexion, abduction, flexion, flexion.
    # The thumb uses opposition followed by three joints restricted here to
    # their nonnegative branch, excluding the old hyperextended thumb pose.
    flexion_indices = np.array([0, 1, 2, 3] + [4*i+j for i in range(1, 5) for j in [0, 2, 3]])

    def __init__(self, model):
        self.model = model
        self.joints = {side: np.array([model.joint(f'{side}_{finger}_joint{i}').id
                                     for finger in FINGERS for i in range(1, 5)])
                       for side in 'lr'}
        self.qindices = {s: model.jnt_qposadr[j] for s, j in self.joints.items()}

    def bounds(self, side):
        limits = self.model.jnt_range[self.joints[side]].copy()
        limits[self.flexion_indices, 0] = np.maximum(limits[self.flexion_indices, 0], 0.)
        return limits[:, 0], limits[:, 1]

    def valid(self, side, hand, tolerance=1e-8):
        lower, upper = self.bounds(side)
        hand = np.asarray(hand)
        return bool(hand.shape == (20,) and np.all(np.isfinite(hand))
                    and np.all(hand >= lower-tolerance) and np.all(hand <= upper+tolerance))

    def open(self, side):
        lower, upper = self.bounds(side)
        return np.clip(np.zeros(20), lower, upper)

    def finger_ik(self, controller, side, finger, target, seed):
        index = FINGERS.index(finger)
        lower, upper = self.bounds(side)
        segment = slice(4*index, 4*index+4)
        lo, hi = lower[segment]+1e-6, upper[segment]-1e-6
        scratch = controller.k
        scratch.qpos[:] = controller.d.qpos
        indices = self.qindices[side][segment]
        site = self.model.site(f'{side}_{finger}_grip').id
        seed = np.clip(seed, lo, hi)

        def residual(q):
            scratch.qpos[indices] = q
            mujoco.mj_kinematics(self.model, scratch)
            return np.r_[100*(scratch.site_xpos[site]-target), .003*(q-seed)]

        result = least_squares(residual, seed, bounds=(lo, hi), max_nfev=150)
        return result.x, float(np.linalg.norm(result.fun[:3])/100)


class NativeHandProfile(NaturalHandProfile):
    """Original robot hand range for the user-prioritized reference motions."""
    def __init__(self,model,open_targets=None):
        super().__init__(model)
        self.open_targets={} if open_targets is None else open_targets

    def bounds(self,side):
        limits=self.model.jnt_range[self.joints[side]].copy()
        return limits[:,0],limits[:,1]

    def open(self,side):
        if side in self.open_targets:
            return np.clip(self.open_targets[side],*self.bounds(side))
        return super().open(side)


class NaturalG2Controller(G2Controller):
    def __init__(self, model, data):
        super().__init__(model, data)
        self.hand_profile = NaturalHandProfile(model)
        self.hand_actuators = {side:np.array([np.flatnonzero(self.actq==q)[0] for q in self.handq[side]])
                               for side in 'lr'}

    def constrain_motor_commands(self):
        # Jacobian force feed-forward also changes a position actuator's
        # effective target. Apply natural bounds to that final command, not
        # only the nominal grasp, without changing the native model limits.
        for side,actuators in self.hand_actuators.items():
            for index,finger in enumerate(FINGERS):
                loaded = np.any(self.tip_forces.get(side+'_'+finger+'_grip',0.))
                flexion = self.hand_profile.flexion_indices[(self.hand_profile.flexion_indices>=4*index)
                                                           &(self.hand_profile.flexion_indices<4*index+4)]
                # Native joints permit extension. Under contact load the
                # measured finger can bend backwards despite a positive
                # target. Resist that motion using the same limited motor,
                # starting before zero; do not alter native model limits.
                q = self.d.qpos[self.handq[side][flexion]]
                # Residual contact/inertia remains after commanded grip force
                # reaches zero. Keep a small extension guard while withdrawing,
                # without curling an otherwise neutral open hand.
                margin = .005 if loaded else -.0025
                self.d.ctrl[actuators[flexion]] += 10.*np.maximum(0.,margin-q)
            self.d.ctrl[actuators] = np.clip(self.d.ctrl[actuators],*self.hand_profile.bounds(side))

    def finger_ik(self, side, finger, point, seed):
        return self.hand_profile.finger_ik(self, side, finger, point, seed)

    def step(self, target):
        for side in 'lr':
            if not self.hand_profile.valid(side, target[self.handq[side]], tolerance=1e-5):
                raise RuntimeError('Command would reverse a finger flexion joint: '+side)
        super().step(target)

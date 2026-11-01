# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Native G2 motor groups and measured contact support for Rubik evaluation.

Extracted from the supplied data-collection physics implementation; collection
planning and expert action generation are not part of this runtime.
"""

import copy
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .control import Controller

FINGERS = ['thumb', 'index_finger', 'middle_finger', 'ring_finger', 'pinky']

NATIVE_FINGERS = ['thumb', 'index', 'middle', 'ring', 'pinky']

def native_tips(m):
    """Identify G2 native pad meshes without changing enabled collision pairs."""
    ids=set()
    for side in 'lr':
        for index, finger in enumerate(FINGERS,1):
            bodies={m.body(f'{side}_{finger}_distal').id:'tip_link',m.body(f'{side}_{finger}_link4').id:'link4'}
            for g in range(m.ngeom):
                b=int(m.geom_bodyid[g])
                if b not in bodies or not(m.geom_contype[g] or m.geom_conaffinity[g]):continue
                expected=f'{"left" if side=="l" else "right"}_finger{index}_{bodies[b]}'
                if m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH or m.mesh(int(m.geom_dataid[g])).name!=expected:
                    raise ValueError('G2 pad must retain its native mesh')
                ids.add(g)
    if len(ids)!=20:raise ValueError('Expected native G2 distal segments and pads')
    return ids

def supporting_fingers(m,d,root,tip_geoms):
    fingers=set();force=np.zeros(6)
    for i,contact in enumerate(d.contact):
        for robot,cube in (contact.geom,contact.geom[::-1]):
            if int(robot) not in tip_geoms or m.geom_bodyid[cube]<root:continue
            mujoco.mj_contactForce(m,d,i,force)
            if force[0]>.05:
                name=m.body(int(m.geom_bodyid[robot])).name
                fingers.add(name.removesuffix('_link4').removesuffix('_distal')+'_distal')
    return fingers

class G2Controller(Controller):
    ik_rotation_tolerance=np.deg2rad(3)
    def __init__(self,m,d):
        super().__init__(m,d)
        if 'GRASP_REFERENCE' in globals():
            self.reference_state=GRASP_REFERENCE[0].copy()
            self.reference_config=copy.deepcopy(GRASP_REFERENCE[1])
        self.home={s:d.qpos[self.armq[s]].copy() for s in 'lr'}
        self.extra_contacts={}

    def step(self,target):
        for name,(side,normal,force) in self.extra_contacts.items():
            R=self.d.xmat[self.m.body(side+'_wrist').id].reshape(3,3)
            self.tip_forces[name]=self.tip_forces.get(name,np.zeros(3))+R@normal*force
        self.transform_tip_forces()
        super().step(target)

    def transform_tip_forces(self):
        pass

    def arm_ik(self,side,pos,rot,seed,relax=False):
        idx=self.armq[side];bounds=self.m.jnt_range[self.armj[side]].copy()
        bounds[:,0]+=.01;bounds[:,1]-=.01
        self.k.qpos[:]=self.d.qpos;body=self.m.body(side+'_wrist').id
        def residual(q):
            self.k.qpos[idx]=q;mujoco.mj_kinematics(self.m,self.k)
            return np.r_[15*(self.k.xpos[body]-pos),
                         Rotation.from_matrix(rot@self.k.xmat[body].reshape(3,3).T).as_rotvec(),
                         .008*(q-seed)]
        result=least_squares(residual,np.clip(seed,bounds[:,0]+1e-5,bounds[:,1]-1e-5),
                             bounds=(bounds[:,0],bounds[:,1]),max_nfev=100,ftol=1e-8,xtol=1e-8,gtol=1e-8)
        self.ik_rotation_error=float(np.linalg.norm(result.fun[3:6]))
        return result.x,float(np.linalg.norm(result.fun[:3])/15)

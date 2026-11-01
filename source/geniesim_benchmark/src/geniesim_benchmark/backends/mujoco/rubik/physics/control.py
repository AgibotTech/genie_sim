# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Bimanual actuator control. IK operates on a scratch MjData only."""
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .posture import LEFT_HOME_DEG,RIGHT_HOME

FINGERS=['thumb','index_finger','middle_finger','ring_finger','pinky']

class Controller:
    def __init__(self,m,d):
        self.m,self.d=m,d
        self.k=mujoco.MjData(m)
        self.armj={s:np.array([m.joint(f'Joint{i}_{s.upper()}').id for i in range(1,8)]) for s in ['l','r']}
        self.handj={s:np.array([j for j in range(m.njnt) if m.joint(j).name.startswith(s+'_')]) for s in ['l','r']}
        self.armq={s:m.jnt_qposadr[j] for s,j in self.armj.items()}
        self.handq={s:m.jnt_qposadr[j] for s,j in self.handj.items()}
        self.home={'l':np.deg2rad(LEFT_HOME_DEG),'r':RIGHT_HOME}
        self.actq=m.jnt_qposadr[m.actuator_trnid[:,0]]
        self.armv=np.concatenate([m.jnt_dofadr[j] for j in self.armj.values()])
        self.command=d.qpos[self.actq].copy()
        self.tip_forces={}
        self.arm_rate=1.2
        self.relaxed_reach=False

    def home_state(self):
        q=self.d.qpos.copy()
        for s in ['l','r']: q[self.armq[s]]=self.home[s]
        return q

    def arm_ik(self,side,pos,rot,seed,relax=False):
        qidx=self.armq[side];j=self.armj[side];bounds=self.m.jnt_range[j].copy()
        wide=relax or self.relaxed_reach
        corridor=(55 if side=='l' else 80) if wide else (40 if side=='l' else 65)
        bounds[:4,0]=np.maximum(bounds[:4,0],self.home[side][:4]-np.deg2rad(corridor))
        bounds[:4,1]=np.minimum(bounds[:4,1],self.home[side][:4]+np.deg2rad(corridor))
        if wide:bounds[3,1]=min(bounds[3,1],np.deg2rad(-5))
        self.k.qpos[:]=self.d.qpos;bid=self.m.body(side+'_wrist').id
        def fun(q):
            self.k.qpos[qidx]=q;mujoco.mj_kinematics(self.m,self.k)
            R=self.k.xmat[bid].reshape(3,3)
            return np.r_[10*(self.k.xpos[bid]-pos),Rotation.from_matrix(rot@R.T).as_rotvec(),.015*(q-seed)]
        result=least_squares(fun,np.clip(seed,bounds[:,0]+1e-5,bounds[:,1]-1e-5),bounds=(bounds[:,0]+1e-5,bounds[:,1]-1e-5),max_nfev=150,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        return result.x, np.linalg.norm(result.fun[:3])/10

    def hand_ik(self,side,targets,seed=None):
        j=self.handj[side];qidx=self.handq[side];w=self.m.body(side+'_wrist').id
        self.k.qpos[:]=self.d.qpos;mujoco.mj_kinematics(self.m,self.k)
        p=self.k.xpos[w].copy();R=self.k.xmat[w].reshape(3,3).copy()
        sites=[self.m.site(side+'_'+f+'_grip').id for f in FINGERS]
        bounds=self.m.jnt_range[j]
        def fun(q):
            self.k.qpos[qidx]=q;mujoco.mj_kinematics(self.m,self.k)
            local=(self.k.site_xpos[sites]-p)@R
            return np.r_[(local-np.asarray(targets)).ravel()*100,.005*q]
        seed=np.tile([.4,0,.7,.6],5) if seed is None else seed
        result=least_squares(fun,np.clip(seed,bounds[:,0]+1e-4,bounds[:,1]-1e-4),bounds=(bounds[:,0]+1e-4,bounds[:,1]-1e-4),max_nfev=200,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        return result.x,result.fun[:15].reshape(5,3)*10

    def finger_ik(self,side,finger,point,seed):
        fi=FINGERS.index(finger);j=self.handj[side][fi*4:fi*4+4];idx=self.handq[side][fi*4:fi*4+4]
        bounds=self.m.jnt_range[j];site=self.m.site(side+'_'+finger+'_grip').id
        self.k.qpos[:]=self.d.qpos
        def fun(q):
            self.k.qpos[idx]=q;mujoco.mj_kinematics(self.m,self.k)
            return np.r_[(self.k.site_xpos[site]-point)*100,.003*(q-seed)]
        result=least_squares(fun,np.clip(seed,bounds[:,0]+1e-5,bounds[:,1]-1e-5),bounds=(bounds[:,0]+1e-5,bounds[:,1]-1e-5),max_nfev=120)
        return result.x,np.linalg.norm(result.fun[:3])/100

    def constrain_motor_commands(self):
        """Optional robot-specific constraints after all feed-forward terms."""
        pass

    def step(self,target):
        desired=target[self.actq]
        maxrate=np.full(len(desired),2.5)
        for q in self.armq.values():maxrate[np.isin(self.actq,q)]=self.arm_rate
        self.command+=np.clip(desired-self.command,-maxrate*self.m.opt.timestep,maxrate*self.m.opt.timestep)
        self.d.ctrl[:]=self.command
        if self.tip_forces:
            torque=np.zeros(self.m.nv)
            for name,force in self.tip_forces.items():
                jac=np.zeros((3,self.m.nv))
                mujoco.mj_jacSite(self.m,self.d,jac,None,self.m.site(name).id)
                torque+=jac.T@force
            for side in ['l','r']:
                for j in self.handj[side]:
                    act=np.flatnonzero(self.m.actuator_trnid[:,0]==j)
                    if len(act):
                        a=act[0];self.d.ctrl[a]+=torque[self.m.jnt_dofadr[j]]/self.m.actuator_gainprm[a,0]
        self.d.qfrc_applied[:]=0
        # Compensation goes through the motor, so its total output obeys the
        # native force limit. No extra generalized robot/cube force is added.
        for a,j in enumerate(self.m.actuator_trnid[:,0]):
            if self.m.jnt_dofadr[j] in self.armv:
                self.d.ctrl[a]+=self.d.qfrc_bias[self.m.jnt_dofadr[j]]/self.m.actuator_gainprm[a,0]
        self.constrain_motor_commands()
        mujoco.mj_step(self.m,self.d)

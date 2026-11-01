# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Cube mechanics metadata and read-only facelet decoding (URFDLB convention)."""
import itertools
import numpy as np
from scipy.spatial.transform import Rotation

SCRAMBLE = "R U F' L2 D B R' U2 F D'"
FACES = 'URFDLB'
NORMALS = np.array([[0,0,1],[1,0,0],[0,-1,0],[0,0,-1],[-1,0,0],[0,1,0]])
RIGHT = np.array([[1,0,0],[0,1,0],[1,0,0],[1,0,0],[0,-1,0],[-1,0,0]])
DOWN = np.array([[0,-1,0],[0,0,-1],[0,0,-1],[0,1,0],[0,0,-1],[0,0,-1]])
COLORS = ['#f4f3e9','#db3737','#25a66f','#ffd34b','#f28a30','#307fd6']
PITCH = .01905

def coords(name):
    return np.array([1 if 'pos_'+a in name else -1 if 'neg_'+a in name else 0 for a in 'xyz'])

def move_rotation(token):
    if token[0] not in FACES or token[1:] not in ['', "'", '2']:
        raise ValueError('Invalid move: '+token)
    n=NORMALS[FACES.index(token[0])]
    angle=-np.pi/2 * (2 if token.endswith('2') else -1 if token.endswith("'") else 1)
    return n, Rotation.from_rotvec(n*angle).as_matrix()

def inverse(sequence):
    return ' '.join(t if t.endswith('2') else t[0] if t.endswith("'") else t+"'" for t in sequence.split()[::-1])

def legal_state(state):
    """Check edge flips, corner twists and permutation parity without solving."""
    if state is None:return None
    if len(state)!=54 or any(state.count(f)!=9 for f in FACES):return False
    from kociemba.pykociemba.facecube import FaceCube
    try:return FaceCube(state).toCubieCube().verify()==0
    except (ValueError,IndexError,AssertionError):return False

def logical_cubies(sequence=''):
    items=[{'c':np.array(c), 'R':np.eye(3)} for c in itertools.product([-1,0,1],repeat=3) if any(c)]
    for token in sequence.split():
        n,rot=move_rotation(token)
        for item in items:
            if np.dot(item['R']@item['c'],n)>.5:
                item['R']=np.rint(rot@item['R'])
    return items

def facelets(items, tolerance_deg=3):
    """Returns None during unaligned rotations; never rounds the running simulation."""
    output=['?']*54
    worst=0.
    invalid=False
    for item in items:
        c,R=item['c'],item['R']
        p=R@c
        for color,n0 in enumerate(NORMALS):
            if np.dot(c,n0)!=1: continue
            n=R@n0
            face=int(np.argmax(NORMALS@n))
            angle=np.degrees(np.arccos(np.clip(np.dot(n,NORMALS[face]),-1,1)))
            worst=max(worst,float(angle))
            if angle>tolerance_deg: invalid=True
            row=int(np.rint(np.dot(p,DOWN[face])))+1
            col=int(np.rint(np.dot(p,RIGHT[face])))+1
            if not(0<=row<3 and 0<=col<3): invalid=True; continue
            index=face*9+row*3+col
            if output[index]!='?': invalid=True
            output[index]=FACES[color]
    if '?' in output or invalid: return None,worst
    return ''.join(output),worst

class CubeState:
    def __init__(self,m):
        self.m=m
        self.root=m.body('cube:middle').id
        self.items=[]
        for b in range(m.nbody):
            name=m.body(b).name
            if name and name.startswith('cube:cubelet:'):
                self.items.append((b,coords(name)))

    def decode(self,d,tolerance_deg=3):
        root=d.xmat[self.root].reshape(3,3)
        return facelets([{'c':c,'R':root.T@d.xmat[b].reshape(3,3)} for b,c in self.items],tolerance_deg)

    def initialize(self,d,sequence=SCRAMBLE):
        """Only legal before mj_step. Execution code must never call this method."""
        if d.time != 0: raise RuntimeError('Cannot initialize a running simulation')
        by_coord={tuple(x['c']):x['R'] for x in logical_cubies(sequence)}
        for b,c in self.items:
            parent=self.m.body_parentid[b]
            adr=self.m.body_jntadr[parent]
            count=self.m.body_jntnum[parent]
            R=by_coord[tuple(c)]
            if np.count_nonzero(c)==1:
                axis=np.flatnonzero(c)[0]
                d.qpos[self.m.jnt_qposadr[adr]]=Rotation.from_matrix(R).as_rotvec()[axis]
            else:
                import mujoco
                if self.m.jnt_type[adr]==mujoco.mjtJoint.mjJNT_BALL:
                    q=self.m.jnt_qposadr[adr]
                    d.qpos[q:q+4]=np.roll(Rotation.from_matrix(R).as_quat(),1)
                else:
                    angles=Rotation.from_matrix(R).as_euler('XYZ')
                    for i in range(min(count,3)):
                        d.qpos[self.m.jnt_qposadr[adr+i]]=angles[i]

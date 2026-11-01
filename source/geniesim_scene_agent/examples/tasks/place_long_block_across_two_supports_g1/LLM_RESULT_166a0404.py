# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random
import math
import numpy as np

CONFIG = {'x': -0.112, 'x_jitter': 0.02, 'y_jitter': 0.025, 'forward': 0.065, 'forward_jitter': 0.012, 'lateral_jitter': 0.02, 'spacing': 0.2, 'spacing_jitter': 0.02, 'beam_yaw': 15, 'support_yaw': 20}

def separated(a,ha,aa,b,hb,ab,margin=.004):
 ra=np.array([[math.cos(aa),-math.sin(aa)],[math.sin(aa),math.cos(aa)]])
 rb=np.array([[math.cos(ab),-math.sin(ab)],[math.sin(ab),math.cos(ab)]])
 axes=np.concatenate([ra.T,rb.T]);delta=np.asarray(b)-a
 return any(abs(ax@delta)>np.abs(ax@ra)@ha+np.abs(ax@rb)@hb+margin for ax in axes)

def sample(rng,c):
 for attempt in range(10000):
  x=c['x']+rng.uniform(-c['x_jitter'],c['x_jitter']);y=rng.uniform(-c['y_jitter'],c['y_jitter'])
  yaw=math.radians(rng.uniform(-c['beam_yaw'],c['beam_yaw']));syaw=math.radians(rng.uniform(-c['support_yaw'],c['support_yaw']))
  forward=c['forward']+rng.uniform(-c['forward_jitter'],c['forward_jitter']);lateral=rng.uniform(-c['lateral_jitter'],c['lateral_jitter'])
  gap=c['spacing']+rng.uniform(-c['spacing_jitter'],c['spacing_jitter']);axis=np.array([-math.sin(syaw),math.cos(syaw)])
  supports=[np.array([x+forward,y+lateral])+sign*gap/2*axis for sign in [1,-1]]
  if all(separated(np.array([x,y]),[.02,.20],yaw,p,[.04,.04],syaw) for p in supports):
   return dict(beam_xyz=[x,y,.754],beam_yaw_rad=yaw,support_xyz=[[*p,.754] for p in supports],support_yaw_rad=syaw,forward_m=forward,lateral_m=lateral,spacing_m=gap,rejection_attempts=attempt)
 raise RuntimeError('Unable to generate nonoverlapping scene')

@register()
def root_scene() -> Shape:
    params=sample(random,CONFIG)
    table=library_call('usd',oid='table_000',keywords=['workspace_table','table'])
    top=float(compute_shape_max(table)[2])
    objects=[table]
    specs=[('benchmark_building_blocks_169',params['beam_xyz'],params['beam_yaw_rad'],['lift_block','yellow','long_block','on_table'])]
    specs += [('benchmark_building_blocks_172',pos,params['support_yaw_rad'],['bridge_support',side,'red','on_table']) for side,pos in zip(['left','right'],params['support_xyz'])]
    for asset,pos,yaw,keywords in specs:
        obj=library_call('usd',oid=asset,keywords=keywords)
        obj=transform_shape(obj,rotation_matrix(yaw,direction=(0,0,1),point=(0,0,0)))
        obj=transform_shape(obj,translation_matrix([pos[0],pos[1],top-float(compute_shape_min(obj)[2])+.002]))
        objects.append(obj)
    return concat_shapes(*objects)

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Explicit, reproducible user-selected surface friction adjustments."""
import numpy as np
import mujoco
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def apply_surface_friction(model, scale=1., cube_sliding_friction=None):
    if not np.isfinite(scale) or not 1. <= scale <= 2.:
        raise ValueError('Surface friction scale must be finite and between one and two')
    if cube_sliding_friction is not None and (not np.isfinite(cube_sliding_friction) or cube_sliding_friction<=0):
        raise ValueError('Cube sliding friction must be finite and positive')
    wrists={body for body in range(model.nbody) if model.body(body).name in ('l_wrist','r_wrist')}
    hands=set(wrists)
    for body in range(1,model.nbody):
        if int(model.body_parentid[body]) in hands:
            hands.add(body)
    changes=[]
    for geom in range(model.ngeom):
        body=int(model.geom_bodyid[geom])
        name=model.body(body).name or ''
        if not (model.geom_contype[geom] | model.geom_conaffinity[geom]):
            continue
        if body not in hands and not name.startswith('cube:cubelet:'):
            continue
        old=float(model.geom_friction[geom,0])
        model.geom_friction[geom,0]=old*scale
        if name.startswith('cube:cubelet:') and cube_sliding_friction is not None:
            model.geom_friction[geom,0]=cube_sliding_friction
        changes.append(dict(geom=geom,body=name,before=old,after=float(model.geom_friction[geom,0])))
    return changes


def with_internal_cube_friction(model, coefficient=.05, *, source_xml=None):
    """Compile internal piece contact pairs while retaining external coatings.

    Call before creating the execution MjData. This returns a new model with
    identical generalized coordinates and robot parameters. Explicit pairs
    separate the lubricated piece interfaces from hand/shell friction.
    """
    if not np.isfinite(coefficient) or coefficient<=0:
        raise ValueError('Internal cube friction must be finite and positive')
    if model.npair:
        raise ValueError('Internal contact conversion expects no existing explicit pairs')
    if source_xml is None:
        raise ValueError('The calibrated MJCF source must be supplied by the asset loader')
    template=mujoco.MjModel.from_xml_string(source_xml)
    for field in ('nbody','njnt','nq','nv','nu','ngeom','nmesh'):
        if getattr(template,field)!=getattr(model,field):
            raise ValueError('Contact-model topology mismatch: '+field)
    for kind,count in (('body',model.nbody),('joint',model.njnt),('actuator',model.nu)):
        if any(getattr(template,kind)(i).name!=getattr(model,kind)(i).name for i in range(count)):
            raise ValueError('Contact-model name ordering mismatch: '+kind)
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'model.xml'
        mujoco.mj_saveLastXML(str(path),model)
        scene=ET.parse(path).getroot()
    names={}
    world=scene.find('worldbody')
    for body in [world,*world.findall('.//body')]:
        body_id=0 if body is world else model.body(body.get('name')).id
        geoms=body.findall('geom')
        if len(geoms)!=model.body_geomnum[body_id]:
            raise ValueError('Contact-model geometry ordering mismatch')
        for offset,geom in enumerate(geoms):
            index=int(model.body_geomadr[body_id])+offset
            name=geom.get('name') or 'native_contact_geom_'+str(index)
            geom.set('name',name);names[index]=name
    contact=scene.find('contact')
    if contact is None:contact=ET.SubElement(scene,'contact')
    shells=[g for g in range(model.ngeom)
            if (model.body(int(model.geom_bodyid[g])).name or '').startswith('cube:cubelet:')
            and (model.geom_contype[g]|model.geom_conaffinity[g])]
    excluded=set(map(int,model.exclude_signature))
    for offset,a in enumerate(shells):
        for b in shells[offset+1:]:
            ba,bb=sorted((int(model.geom_bodyid[a]),int(model.geom_bodyid[b])))
            if ba==bb or ((ba<<16)|bb) in excluded:
                continue
            if not ((model.geom_contype[a]&model.geom_conaffinity[b]) or
                    (model.geom_contype[b]&model.geom_conaffinity[a])):
                continue
            wa,wb=int(model.body_weldid[ba]),int(model.body_weldid[bb])
            if wa==wb or int(model.body_parentid[wa])==wb or int(model.body_parentid[wb])==wa:
                continue
            friction=np.maximum(model.geom_friction[a],model.geom_friction[b])
            values=[coefficient,coefficient,friction[1],friction[2],friction[2]]
            ET.SubElement(contact,'pair',name=f'internal_cube_{a}_{b}',geom1=names[a],geom2=names[b],
                condim=str(max(int(model.geom_condim[a]),int(model.geom_condim[b]))),
                friction=' '.join(format(x,'.17g') for x in values),
                solref=' '.join(format(x,'.17g') for x in model.geom_solref[a]),
                solimp=' '.join(format(x,'.17g') for x in model.geom_solimp[a]))
    updated=mujoco.MjModel.from_xml_string(ET.tostring(scene,encoding='unicode'))
    # XML shorthand/defaults and text precision can otherwise change gains,
    # calibrated sites and fixed-body quaternions. Restore numeric arrays;
    # topology and mesh indexing were checked above and do not change here.
    for name in dir(model):
        if name.startswith(('_','pair_')):
            continue
        value=getattr(model,name)
        if isinstance(value,np.ndarray) and value.dtype.kind=='f':
            target=getattr(updated,name)
            if target.shape!=value.shape:
                raise ValueError('Unexpected numeric model shape change: '+name)
            target[:]=value
    for name in ('geom_contype','geom_conaffinity','geom_condim','geom_priority','geom_group',
                 'actuator_forcelimited','actuator_ctrllimited','actuator_dyntype','actuator_gaintype','actuator_biastype'):
        getattr(updated,name)[:]=getattr(model,name)
    for name in dir(model.opt):
        if name.startswith('_'):
            continue
        value=getattr(model.opt,name)
        if callable(value):continue
        if isinstance(value,np.ndarray):getattr(updated.opt,name)[:]=value
        else:setattr(updated.opt,name,value)
    return updated

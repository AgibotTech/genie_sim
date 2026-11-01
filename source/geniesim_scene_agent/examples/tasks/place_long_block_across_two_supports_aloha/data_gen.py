# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Physically grasp both sides and lift together, following lift_pot_g2's dual APIs."""
import os
import json
import numpy as np
from common.base_utils.transform_utils import quat2mat_wxyz
from client.llm_gen.robot_api import (RobotAPI, GraspConfig, MoveConfig, DualMoveToPose,
                                     BimanualRigidMoveConfig)

ROBOT = 'aloha'
IS_OP = ROBOT in ('g1_op', 'g2_op')
LIFT_HEIGHT = 0.12
END_OFFSETS = (0.145, 0.15, 0.155)


def object_matrix(robot, actor):
    p, q = robot.get_object_pose(actor)
    pose = np.eye(4)
    pose[:3, :3] = quat2mat_wxyz(q)
    pose[:3, 3] = p
    return pose


def grasp_poses(pose, arm):
    long = pose[:3, 2].copy()
    if long[1] < 0:
        long *= -1
    down = np.array([0., 0., -1.])
    across = np.cross(long, -down)
    across /= np.linalg.norm(across)
    if IS_OP:
        across *= -1
    sign = 1 if arm == 'left' else -1
    angle = np.deg2rad(55 if IS_OP else 0)
    approach = np.cos(angle)*down - sign*np.sin(angle)*long
    lateral = np.cross(approach, across)
    lateral /= np.linalg.norm(lateral)
    rotation = np.column_stack((across, lateral, np.cross(across, lateral)))
    poses = []
    for distance in END_OFFSETS:
        p = np.eye(4)
        p[:3, :3] = rotation
        p[:3, 3] = pose[:3, 3] + sign*distance*long
        p[2, 3] += .012 if IS_OP else 0.
        poses.append(p)
    cfg = GraspConfig(disable_upside_down=IS_OP, flip_grasp=False,
                      grasp_pose=np.asarray(poses), grasp_pose_frame='world')
    poses = cfg.grasp_pose
    if cfg.disable_upside_down:
        poses = poses[poses[:, 2, 1] < 0 if arm == 'left' else poses[:, 2, 1] > 0]
    return poses


def bottom(pose):
    # Payload local axes: X/Y are 4 cm; Z is the 40 cm long axis.
    return pose[2, 3] - np.abs(pose[2, :3]) @ np.array([.02, .02, .20])


def hold_checks(robot, actor, initial, arms):
    start = object_matrix(robot, actor)
    relative = {a: np.linalg.inv(robot.get_ee_pose(a)) @ start for a in arms}
    observations = []
    for _ in range(10):
        robot.wait(.2)
        current = object_matrix(robot, actor)
        tilt = float(np.degrees(np.arcsin(np.clip(abs(current[2, 2]), 0, 1))))
        slip, turn = 0., 0.
        for arm in arms:
            rel = np.linalg.inv(robot.get_ee_pose(arm)) @ current
            slip = max(slip, float(np.linalg.norm(rel[:3,3]-relative[arm][:3,3])))
            turn = max(turn, float(np.degrees(np.arccos(np.clip((np.trace(relative[arm][:3,:3].T@rel[:3,:3])-1)/2,-1,1)))))
        observations.append(dict(bottom_rise=float(bottom(current)-bottom(initial)), tilt_deg=tilt,
                                 slip_m=slip, rotation_deg=turn))
    passed = all(x['bottom_rise'] >= .08 and x['tilt_deg'] <= 8 and x['slip_m'] <= .005
                 and x['rotation_deg'] <= 5 for x in observations)
    print('LIFT_PHYSICS', json.dumps(dict(mode=os.environ.get('LIFT_TEST_MODE','both'),
          passed=passed, samples=observations)), flush=True)
    return passed


def dual_linear(robot, targets, block, description):
    # Use the supported rigid dual planner for synchronized vertical translation.
    return robot.bimanual_rigid_move(np.repeat(targets['right'][None],3,axis=0),
        config=BimanualRigidMoveConfig(leader_arm='right', max_consecutive_failures=1,
            path_constraint=[], offset_and_constraint_in_goal_frame=False,
            action_description={'zh':description,'en':'moves both hands vertically together'}))


def place_on_supports(robot, block, table):
    supports = [k for k,v in robot.scene_info['layout'].items() if 'bridge_support' in v.get('keywords',[])]
    if len(supports)!=2:return False
    bases = [object_matrix(robot,k) for k in supports]
    held = object_matrix(robot,block)
    center = (bases[0][:3,3]+bases[1][:3,3])/2
    top = max(p[2,3]+robot.get_object_size_in_world_frame(k)[2]/2 for p,k in zip(bases,supports))
    desired = held.copy()
    # Both foundations share the same orientation and the beam spans their centers.
    desired[:3,:3] = bases[0][:3,:3]
    desired[:3,3] = center
    desired[2,3] = top+.02+.025
    relative = {a:np.linalg.inv(held)@robot.get_ee_pose(a) for a in ('left','right')}
    target = desired@relative['right']
    if not robot.bimanual_rigid_move(np.repeat(target[None],3,axis=0),config=BimanualRigidMoveConfig(
        leader_arm='right',max_consecutive_failures=1,unmovable_objects=[table]+supports,
        action_description={'zh':'将长条积木移到两块地基上方','en':'carries the beam above both supports'})):
        print('BRIDGE_FAILED transfer',flush=True);return False
    actual = object_matrix(robot,block)
    landing = desired.copy();landing[2,3]=top+.02+.002
    delta = landing@np.linalg.inv(actual)
    targets = {a:delta@robot.get_ee_pose(a) for a in ('left','right')}
    if not dual_linear(robot,targets,block,'双手缓慢将积木落到两块地基上'):
        print('BRIDGE_FAILED lower',flush=True);return False
    robot.open_gripper(arm='left');robot.open_gripper(arm='right')
    robot.client.detach_obj()
    # After release the arms are independent; avoid preserving contact-induced wrist tilt.
    for arm, sign in [('left',1),('right',-1)]:
        current=robot.get_ee_pose(arm)
        candidates=[]
        for dx in (0.,-.025,-.05):
            for dy in (0.,.03,.06):
                for dz in (.035,.05):
                    q=current.copy()
                    q[:3,:3]=grasp_poses(bases[0],arm)[0,:3,:3]
                    q[:3,3]+=[dx,sign*dy,dz]
                    candidates.append(q)
        if not robot.move_actor_to_target_pose('gripper',np.asarray(candidates),arm=arm,
                move_config=MoveConfig(unmovable_objects=[table,block]+supports,
                    action_description={'zh':'松手后撤离积木','en':'withdraws from the placed beam'})):
            print('BRIDGE_FAILED retreat',arm,flush=True);return False
    robot.wait(.5)
    start=object_matrix(robot,block)
    samples=[]
    for _ in range(10):
        robot.wait(.2);p=object_matrix(robot,block)
        # Each support center must lie beneath the beam footprint, with margin.
        inv=np.linalg.inv(p);under=[inv@np.r_[object_matrix(robot,k)[:3,3],1.] for k in supports]
        footprint=all(abs(q[0])<.025 and abs(q[2])<.17 for q in under)
        tilt=float(np.degrees(np.arcsin(np.clip(abs(p[2,2]),0,1))))
        heights=[object_matrix(robot,k)[2,3]+robot.get_object_size_in_world_frame(k)[2]/2 for k in supports]
        gap=float(bottom(p)-sum(heights)/2)
        shift=float(np.linalg.norm(p[:3,3]-start[:3,3]))
        bases_stable=all(np.linalg.norm(object_matrix(robot,k)[:3,3]-b[:3,3])<.01 for k,b in zip(supports,bases))
        samples.append(dict(footprint=footprint,tilt_deg=tilt,gap_m=gap,shift_m=shift,bases_stable=bases_stable))
    passed=all(s['footprint'] and s['tilt_deg']<5 and -.005<s['gap_m']<.008 and s['shift_m']<.005 and s['bases_stable'] for s in samples)
    print('BRIDGE_PHYSICS',json.dumps(dict(passed=passed,samples=samples)),flush=True)
    return passed


def execute_task(robot: RobotAPI):
    mode = 'both'
    arms = ('left','right') if mode == 'both' else (mode,)
    robot.client.set_task_task_basic_info(dict(task_name='双手抬起长条积木并搭放到前方两块地基上',
        english_task_name='Lift the long block and place it across the two supports',
        init_scene_text='A yellow long block lies on the table, with two aligned square supports farther from the robot.'))
    layout = robot.scene_info['layout']
    block = next(k for k,v in layout.items() if 'lift_block' in v.get('keywords',[]))
    table = next(k for k,v in layout.items() if 'workspace_table' in v.get('keywords',[]))
    initial = object_matrix(robot, block)
    grasps = {a: grasp_poses(initial,a) for a in arms}
    pre = {a: p.copy() for a,p in grasps.items()}
    for p in pre.values():
        p[:, 2, 3] += .08
    def move(poses, close=False):
        cfg = lambda: MoveConfig(gripper_action='close' if close else None, unmovable_objects=[table],
            action_description={'zh':'从两侧夹住积木' if close else '移动到积木两侧上方',
                                'en':'grasps the sides of the block' if close else 'approaches above the block'},
            extra_params={'disable_upside_down':IS_OP})
        if mode == 'both':
            return robot.dual_act(left=DualMoveToPose(actor='gripper', target_pose=poses['left'],move_config=cfg()),
                right=DualMoveToPose(actor='gripper',target_pose=poses['right'],move_config=cfg()),
                action_description={'zh':'双手同步夹住积木' if close else '双手同步接近积木',
                                    'en':'both hands grasp the block' if close else 'both hands approach the block'})[0]
        return robot.move_actor_to_target_pose('gripper',poses[mode],arm=mode,move_config=cfg())
    if not move(pre):
        print('LIFT_SETUP_FAILED pregrasp', flush=True)
        return False
    # Only the target is removed from planning obstacles. Physical contact stays enabled.
    robot.remove_objs_from_obstacle([robot.get_object_prim_path(block)])
    if not move(grasps,close=True):
        print('LIFT_SETUP_FAILED grasp', flush=True)
        return False
    print('BOTH_GRASPED' if mode=='both' else 'SINGLE_GRASPED', flush=True)
    for a in arms:
        robot.attach_obj([robot.get_object_prim_path(block)],arm=a)
    before = {a:robot.get_ee_pose(a).copy() for a in arms}
    if mode == 'both':
        targets = np.repeat(before['right'][None],3,axis=0)
        targets[:, 2, 3] += [LIFT_HEIGHT-.005,LIFT_HEIGHT,LIFT_HEIGHT+.005]
        ok = robot.bimanual_rigid_move(targets,config=BimanualRigidMoveConfig(leader_arm='right',
            max_consecutive_failures=1, unmovable_objects=[table],
            action_description={'zh':'双手保持水平共同抬起积木','en':'both hands lift the block together'},
            path_constraint=[.1,.1,.1,.1,.1,0.],offset_and_constraint_in_goal_frame=False))
    else:
        target = before[mode].copy()
        target[2,3] += LIFT_HEIGHT
        robot.client.set_frame_state('single_lift_control',0,'gripper',block,True,arm=mode,target_pose=target)
        ok = robot.robot.move_pose(target,type='Straight',arm=mode)
    rises = {a:float(robot.get_ee_pose(a)[2,3]-before[a][2,3]) for a in arms}
    print('LIFT_COMMAND',json.dumps(dict(mode=mode,motion_ok=bool(ok),ee_rise=rises)),flush=True)
    physical = hold_checks(robot,block,initial,arms)
    return bool(ok) and physical and place_on_supports(robot,block,table)

"""Create a native USD fixed joint at the current pose of two rigid bodies."""

from pxr import Gf, Sdf, UsdGeom, UsdPhysics


def create_fix_joint(stage, body0_path, body1_path, joint_path,
                     collision_enabled=True):
    paths = (body0_path, body1_path, joint_path)
    if any(not isinstance(path, str) or not Sdf.Path(path).IsAbsolutePath()
           or not Sdf.Path(path).IsPrimPath() for path in paths):
        raise ValueError('All paths must be absolute USD prim paths')
    if body0_path == body1_path:
        raise ValueError('A fixed joint requires two different bodies')
    if joint_path in (body0_path, body1_path):
        raise ValueError('Joint path must differ from body paths')
    if stage.GetPrimAtPath(joint_path):
        raise ValueError(f'Joint path already exists: {joint_path}')

    bodies = [stage.GetPrimAtPath(path) for path in (body0_path, body1_path)]
    if not all(prim and prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in bodies):
        raise ValueError('Both joint bodies must exist and have RigidBodyAPI')

    cache = UsdGeom.XformCache()
    body0_world, body1_world = (cache.GetLocalToWorldTransform(prim)
                                for prim in bodies)
    relative = body1_world * body0_world.GetInverse()
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(relative.ExtractTranslation()))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(relative.ExtractRotationQuat()))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1))
    joint.CreateJointEnabledAttr(True)
    joint.CreateCollisionEnabledAttr(bool(collision_enabled))
    return {'success': True, 'path': joint_path,
            'body0': body0_path, 'body1': body1_path}


def get_fix_joints(stage):
    """Return enabled fixed joints and their connected rigid-body paths."""
    result = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.FixedJoint):
            continue
        joint = UsdPhysics.FixedJoint(prim)
        if joint.GetJointEnabledAttr().Get() is False:
            continue
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        if body0 and body1:
            result.append({'path': str(prim.GetPath()),
                           'body0': str(body0[0]), 'body1': str(body1[0])})
    return result

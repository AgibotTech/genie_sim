# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Build CPU-only robot collision filters from topology and authored exclusions."""

from itertools import combinations, product


def enable_robot_self_collision(builder, source_stage=None, excludes=()):
    """Replace blanket robot filters while retaining mechanical and USD exclusions.

    Operate only on the CPU copy, before MuJoCo export computes body excludes
    and contact bitmasks. Fixed-connected bodies form one collision component;
    joints that filter their parent also filter the connected components.
    """
    import newton

    roots = list(range(builder.body_count))

    def root(body):
        while roots[body] != body:
            roots[body] = roots[roots[body]]
            body = roots[body]
        return body

    for kind, parent, child in zip(builder.joint_type, builder.joint_parent, builder.joint_child):
        if kind == newton.JointType.FIXED and parent >= 0:
            roots[root(child)] = root(parent)
    adjacent = {
        tuple(sorted((root(parent), root(child))))
        for parent, child, filtered in zip(
            builder.joint_parent,
            builder.joint_child,
            builder.joint_collision_filter_parent,
        )
        if parent >= 0 and filtered
    }
    colliders = {
        shape
        for shape, body in enumerate(builder.shape_body)
        if body >= 0
        and builder.shape_flags[shape] & newton.ShapeFlags.COLLIDE_SHAPES
        and builder.shape_collision_group[shape] != 0
    }
    # Preserve filters involving disabled shapes or world geometry. Only the
    # robot's active shape pairs are reconstructed from explicit provenance.
    filters = {tuple(sorted(pair)) for pair in builder.shape_collision_filter_pairs if not set(pair) <= colliders}
    for a, b in combinations(sorted(colliders), 2):
        ra, rb = root(builder.shape_body[a]), root(builder.shape_body[b])
        if ra == rb or tuple(sorted((ra, rb))) in adjacent:
            filters.add((a, b))

    if source_stage is not None:
        from pxr import Usd

        def shapes_at(path):
            path = str(path)
            # A nested rigid body is not part of its parent's collision shape.
            # Match Newton's collider-first resolution for dual body/collider prims.
            shapes = {
                i for i, label in enumerate(builder.shape_label) if builder.shape_body[i] >= 0 and str(label) == path
            }
            if shapes:
                return shapes
            return {
                i for i, body in enumerate(builder.shape_body) if body >= 0 and str(builder.body_label[body]) == path
            }

        for prim in Usd.PrimRange.Stage(source_stage, Usd.TraverseInstanceProxies()):
            targets = prim.GetRelationship("physics:filteredPairs").GetTargets()
            if not targets:
                continue
            first = shapes_at(prim.GetPath())
            for target in targets:
                for a, b in product(first, shapes_at(target)):
                    if a != b:
                        filters.add(tuple(sorted((a, b))))

    def body_named(name):
        matches = [
            i
            for i, label in enumerate(builder.body_label)
            if str(label) == name or ("/" not in name and str(label).rsplit("/", 1)[-1] == name)
        ]
        if len(matches) != 1:
            raise ValueError(f"CPU self-collision body {name!r} is missing or ambiguous")
        return matches[0]

    for first, second in excludes:
        a_body, b_body = body_named(first), body_named(second)
        if a_body == b_body:
            raise ValueError("CPU self-collision exclusion must name two distinct bodies")
        for a, b in product(builder.body_shapes[a_body], builder.body_shapes[b_body]):
            filters.add(tuple(sorted((a, b))))
    builder.shape_collision_filter_pairs = sorted(filters)

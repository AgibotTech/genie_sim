# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Keep explicit physics stepping independent from rendering demand."""


def step_world(world, render, timeline):
    # Kit's render=True path only calls app.update() when both dt values
    # are nonzero. Path tracing can leave timeline auto-update disabled;
    # app.update() then renders forever without a physics callback. Explicit
    # stepping still advances PhysX in that state. render() refreshes Fabric
    # and cameras with playSimulations disabled, so it cannot double-step.
    if render and world.is_playing() and world.get_physics_dt() > 0 and not timeline.is_auto_updating():
        world.step(render=False)
        world.render()
    else:
        world.step(render=render)

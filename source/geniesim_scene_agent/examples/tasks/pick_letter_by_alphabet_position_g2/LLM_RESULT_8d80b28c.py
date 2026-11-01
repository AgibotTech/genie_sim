# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from helper import *
import random
import numpy as np

"""
scene_name: table_with_digit_and_letters
description: A table with one digit block (1-9) and 3-4 uppercase letter blocks including the Nth letter.
"""

DIGIT_TO_ASSET = {i: f"benchmark_building_blocks_{34+i:03d}" for i in range(10)}
UPPERCASE_LETTERS = {chr(65+i): f"benchmark_letter_{26+i:03d}" for i in range(26)}


@register()
def table_with_digit_and_letters() -> Shape:
    table_shape = library_call("usd", oid="table_000", keywords=["table", "furniture"])
    table_info = get_object_info(table_shape)
    desktop_sub = get_subpart_info(object_id="table_000", subpart_id="desktop")
    desktop_top_z = table_info["center"][2] + desktop_sub["xyz_max"][2]
    tc = table_info["center"]

    n = random.choice([1, 2, 3, 4, 5, 7, 8])
    target_letter = chr(64 + n)

    digit_oid = DIGIT_TO_ASSET[n]
    digit_shape = library_call("usd", oid=digit_oid, keywords=[f"digit_{n}", "digit_block", "number_hint", "left"])
    digit_bottom = -compute_shape_min(digit_shape)[2]
    dx, dy = random.uniform(-0.1, -0.03), random.uniform(0.08, 0.15)
    digit_placed = transform_shape(digit_shape, translation_matrix([tc[0]+dx, tc[1]+dy, desktop_top_z + digit_bottom]))
    rot = random.uniform(160.0/180*np.pi, 200.0/180*np.pi)
    digit_placed = transform_shape(digit_placed, rotation_matrix(rot, (0,0,1), [tc[0]+dx, tc[1]+dy, desktop_top_z]))

    remaining = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c != target_letter]
    distractors = random.sample(remaining, random.randint(2, 3))
    all_letters = [target_letter] + distractors
    random.shuffle(all_letters)

    placed = [(dx, dy, 0.03, 0.03)]
    letter_shapes = []

    for idx, letter in enumerate(all_letters):
        oid = UPPERCASE_LETTERS[letter]
        is_target = (letter == target_letter)
        side_tag = "right" if idx % 2 == 0 else "left"
        kws = [f"letter_{letter}", "letter_block", "uppercase", side_tag]
        if is_target:
            kws.append("target")

        block = library_call("usd", oid=oid, keywords=kws)
        bi = get_object_info(block)
        hx, hy = bi["size"][0]/2, bi["size"][1]/2
        for _ in range(100):
            cx = random.uniform(-0.1, 0.1)
            cy = random.uniform(-0.18, 0.02)
            ok = all(abs(cx-px) >= (hx+phx+0.03) or abs(cy-py) >= (hy+phy+0.03) for px,py,phx,phy in placed)
            if ok:
                placed.append((cx, cy, hx, hy))
                t = transform_shape(block, translation_matrix([tc[0]+cx, tc[1]+cy, desktop_top_z]))
                r = random.uniform(150.0/180*np.pi, 210.0/180*np.pi)
                letter_shapes.append(transform_shape(t, rotation_matrix(r, (0,0,1), [tc[0]+cx, tc[1]+cy, desktop_top_z])))
                break

    return concat_shapes(table_shape, digit_placed, *letter_shapes)


@register()
def root_scene() -> Shape:
    return library_call("table_with_digit_and_letters")

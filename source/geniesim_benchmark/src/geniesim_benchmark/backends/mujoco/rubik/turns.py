# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Recognize measured single-layer quarter turns in URFDLB facelet order."""


def quarter_turn_neighbors(state):
    """Return the 12 outer-layer quarter-turn states, without solving a cube.

    Native decoding expresses stickers relative to the core, so translating or
    rotating the complete object in the world leaves this state unchanged.
    Center changes, half turns and arbitrary rearrangements are not quarter turns.
    Kociemba is already pinned by the supplied physics package for state legality.
    """
    from kociemba.pykociemba.cubiecube import CubieCube, moveCube
    from kociemba.pykociemba.facecube import FaceCube

    faces = "URFDLB"
    if (
        not isinstance(state, str)
        or len(state) != 54
        or any(state.count(face) != 9 for face in faces)
        or state[4::9] != faces
    ):
        return {}
    cube = FaceCube(state).toCubieCube()
    if cube.verify() != 0:
        return {}
    neighbors = {}
    for index, face in enumerate(faces):
        candidate = CubieCube(cp=cube.cp[:], co=cube.co[:], ep=cube.ep[:], eo=cube.eo[:])
        for count in range(1, 4):
            candidate.multiply(moveCube[index])
            if count != 2:
                neighbors[candidate.toFaceCube().to_String()] = face + ("'" if count == 3 else "")
    return neighbors

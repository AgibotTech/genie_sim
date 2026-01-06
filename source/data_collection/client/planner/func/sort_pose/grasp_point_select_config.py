# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

RIGHT_HAND_Z_AZIMUTH_TABLE_PICK_SCORE_CONFIG = [
    # Right hand z-axis horizontal score configuration
    # For table pickup action
    # Facing left (70-100) has lowest cost, 0,
    # Facing right (100-270) cost gradually increases,
    # Facing forward (270-360 - 70) cost gradually decreases,
    {
        "range": (70, 100),
        "type": "linear",
        "weight": 0.0,
        "offset": 0.0,
    },
    {
        "range": (0, 70),
        "type": "linear",
        "weight": -0.5,
    },
    {
        "range": (100, 270),
        "type": "linear",
        "weight": 1.0,
    },
    {
        "range": (270, 360),
        "type": "linear",
        "weight": -1.5,
    },
]

RIGHT_HAND_Z_ELEVATION_TABLE_PICK_SCORE_CONFIG = [
    # Right hand z-axis vertical score configuration
    # For table pickup action
    # Slightly upward or horizontal (-45～5) has lowest cost, 0,
    # Upward (-45 ～ -90) cost gradually increases,
    # Downward (5～90) cost gradually increases,
    # Backward (90～180) cost gradually decreases,
    {
        "range": (-45, 5),
        "type": "linear",
        "weight": 0.0,
        "offset": 0.0,
    },
    {
        "range": (-90, -45),
        "type": "linear",
        "weight": -0.5,
    },
    {
        "range": (5, 90),
        "type": "linear",
        "weight": 0.5,
    },
]


LEFT_HAND_Z_AZIMUTH_TABLE_PICK_SCORE_CONFIG = [
    # Left hand z-axis horizontal score configuration, symmetric with right hand
    {
        "range": (260, 290),
        "type": "linear",
        "weight": 0.0,
        "offset": 0.0,
    },
    {
        "range": (290, 360),
        "type": "linear",
        "weight": 0.5,
    },
    {
        "range": (90, 260),
        "type": "linear",
        "weight": -1.0,
    },
    {
        "range": (0, 90),
        "type": "linear",
        "weight": 1.5,
    },
]

LEFT_HAND_Z_ELEVATION_TABLE_PICK_SCORE_CONFIG = [
    # Left hand z-axis vertical score configuration, same as right hand
    {
        "range": (-45, 5),
        "type": "linear",
        "weight": 0.0,
        "offset": 0.0,
    },
    {
        "range": (-90, -45),
        "type": "linear",
        "weight": -0.5,
    },
    {
        "range": (5, 90),
        "type": "linear",
        "weight": 0.5,
    },
]

RIGHT_HAND_Z_ELEVATION_TABLE_PICK_FROM_UP_SIDE_SCORE_CONFIG = [
    {
        "range": (-90, 90),
        "type": "linear",
        "weight": 0.5,
    }
]

LEFT_HAND_Z_ELEVATION_TABLE_PICK_FROM_UP_SIDE_SCORE_CONFIG = [
    {
        "range": (-90, 90),
        "type": "linear",
        "weight": 0.5,
    }
]
#     {
#         "range": (90, float("inf")),
#         "type": "quadratic",
#         "weight": 1.0,
#     },
# ]

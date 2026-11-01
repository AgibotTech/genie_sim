# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from .sandbox_tasks import SANDBOX_TASKS

G1_DEFAULT_STATES = {
    "body_state": [0.0, 0.4363, 0.8727, 0.42],
    "init_arm": [
        -1.0751,
        0.6109,
        0.2793,
        -1.2846,
        0.7295,
        1.4957,
        -0.1868,
        1.0734,
        -0.6109,
        -0.2793,
        1.2846,
        -0.7313,
        -1.4957,
        0.1868,
    ],
    "init_hand": [0.0, 0.0],
}

G2_DEFAULT_STATES = {
    "body_state": [-0.83423, 1.2172, 0.10025, 0.0, 0.0],
    "head_state": [0.0, 0.0, 0.11464],
    "init_arm": [
        0.739033,
        -0.717023,
        -1.524419,
        -1.537612,
        0.27811,
        -0.925845,
        -0.839257,
        -0.739033,
        -0.717023,
        1.524419,
        -1.537612,
        -0.27811,
        -0.925845,
        0.839257,
    ],
    "init_hand": [0.785, 0.785],
}

# s2r suite on G2: the real robot's rest pose. Body/head/arm match
# G2_DEFAULT_STATES; only the grippers start wider (0.85 rad).
G2_S2R_STATES = {
    **G2_DEFAULT_STATES,
    "init_hand": [0.85, 0.85],
}

# pack_in_supermarket on G2 (unseen robot): raise the waist and pitch the torso/head
# further down so the arms can reach objects on the tabletop.
G2_PACK_STATES = {
    **G2_DEFAULT_STATES,
    "body_state": [-0.83423, 1.40, -0.13, 0.0, 0.0],
    "head_state": [0.0, 0.0, 0.20],
}

# Mean across the 20 stack_shaped_blocks_by_demo_v2_eval_seen instances
# (instruction_<uuid>.json.robot_state_frame0). Small generalization noise
# was averaged out so the benchmark uses one canonical init pose.
G2_STACK_SHAPED_V2_EVAL_SEEN_STATES = {
    "body_state": [-0.83455, 1.21614, 0.10025, 0.0, 0.0],
    "head_state": [0.0, 0.0, 0.11464],
    "init_arm": [
        0.74104,
        -0.71697,
        -1.52728,
        -1.53719,
        0.27701,
        -0.9224,
        -0.83695,
        -0.74,
        -0.71803,
        1.52301,
        -1.53623,
        -0.27352,
        -0.9251,
        0.83426,
    ],
    "init_hand": [0.7854, 0.7854],
}

# Same averaging procedure for the unseen split (20 instances).
G2_STACK_SHAPED_V2_EVAL_UNSEEN_STATES = {
    "body_state": [-0.83426, 1.21682, 0.10025, 0.0, 0.0],
    "head_state": [0.0, 0.0, 0.11464],
    "init_arm": [
        0.73872,
        -0.72061,
        -1.52563,
        -1.5362,
        0.27792,
        -0.92662,
        -0.83734,
        -0.73798,
        -0.71718,
        1.52535,
        -1.53804,
        -0.27477,
        -0.925,
        0.8383,
    ],
    "init_hand": [0.7854, 0.7854],
}

G2_90D_DEFAULT_STATES = {
    "body_state": [-0.8029, 1.5048, -0.3419, 0.0, 0.0],
    "head_state": [0.0, -0.001, 0.3449],
    "init_arm": [
        0.5264,
        -0.5189,
        -1.0938,
        -1.5954,
        -1.1488,
        0.3386,
        -1.1805,
        -0.5252,
        -0.5194,
        1.0943,
        -1.5958,
        1.1483,
        0.3393,
        1.1809,
    ],
    "init_hand": [0.0, 0.0],
}

G2_90D_HOME_00_STATES = {
    "body_state": [-0.949775, 1.684653, -0.639591, 0.0, 0.0],
    "head_state": [0.0, 0.0, 0.449043],
    # Arms hand-tuned in assets/robot/G2_90d/robot_pick_home_tableware.usda.
    "init_arm": [
        # left arm
        0.764454,
        -0.644027,
        -0.996583,
        -1.335177,
        -1.073377,
        0.036652,
        -1.427679,
        # right arm
        -0.937242,
        -1.158899,
        1.445133,
        -1.417207,
        0.439823,
        0.226893,
        1.390290,
    ],
    "init_hand": [0.0, 0.0],
}

TAKE_CUPS_CABINET_G2_90D_STATES = {
    "body_state": [-0.81594142531, 1.3138985932, -0.34662238945, 0.0, 0.0],
    "head_state": [0.0, 0.0, 0.0],
    "init_arm": [
        # left arm
        0.5263913024,
        -0.51888638662,
        -1.0937978422,
        -1.5954054692,
        -1.1487757137,
        0.33859387489,
        -1.180540706,
        # right arm
        -0.52516957193,
        -0.51940998539,
        1.094321441,
        -1.5957545351,
        1.1482521149,
        0.33929200659,
        1.1808897719,
    ],
    "init_hand": [0.0, 0.0],
}

OPEN_THE_DOOR_G2_90D_STATES = {
    "body_state": [-0.6981123166979405, 1.3523379066598955, -0.6, 0, 0],
    "head_state": [0, 0, 0.34921301919614245],
    "init_arm": [
        # left arm
        1.7323838256725361,
        -1.4511636983067924,
        -1.5990512941697577,
        -1.8002556445282796,
        -0.2202215732473698,
        0.004033890103143022,
        0.0007744206133841417,
        # right arm
        -1.7281721695666301,
        -1.4470093667433506,
        1.6010198229527115,
        -1.79933501637105,
        0.2221582239920755,
        -0.001358531735271221,
        -0.0006869357715750387,
    ],
    "init_hand": [0.0, 0.0],
}

PUSH_THE_CHAIR_G2_90D_STATES = {
    "body_state": [-0.81594142531, 1.5138985932, -0.34662238945, 0, 0],
    "head_state": [0.0, 0.0, 0.34138],
    "init_arm": [
        # left arm
        0.5263913024,
        -0.51888638662,
        -1.0937978422,
        -1.5954054692,
        -1.1487757137,
        0.33859387489,
        -1.180540706,
        # right arm
        -0.52516957193,
        -0.51940998539,
        1.094321441,
        -1.5957545351,
        1.1482521149,
        0.33929200659,
        1.1808897719,
    ],
    "init_hand": [0.0, 0.0],
}

SENDING_FOOD_G2_90D_STATES = {
    "body_state": [-1.0, 2.3385, -1.254, 0, 0],
    "head_state": [0, 0, 0.2],
    "init_arm": [
        # left arm
        1.7323838256725361,
        -1.4511636983067924,
        -1.5990512941697577,
        -1.8002556445282796,
        -0.2202215732473698,
        0.004033890103143022,
        0.0007744206133841417,
        # right arm
        -1.7281721695666301,
        -1.4470093667433506,
        1.6010198229527115,
        -1.79933501637105,
        0.2221582239920755,
        -0.001358531735271221,
        -0.0006869357715750387,
    ],
    "init_hand": [0.0, 0.0],
}

G2_STATES_WBC = {
    "body_state": [-0.81594142531, 1.5138985932, -0.14662238945, 0.0, 0.0],
    "head_state": [0.0, 0.0, 0.45138],
    "init_arm": [
        # left arm
        0.5263913024,
        -0.51888638662,
        -1.0937978422,
        -1.5954054692,
        -1.1487757137,
        0.33859387489,
        -1.180540706,
        # right arm
        -0.52516957193,
        -0.51940998539,
        1.094321441,
        -1.5957545351,
        1.1482521149,
        0.33929200659,
        1.1808897719,
    ],
    "init_hand": [0.0, 0.0],
}

MATCH_AND_PICK_FROM_CONVEYOR_G2_90D_STATES = {
    "body_state": [-0.83423, 1.2172, 0.10025, 0.0, 0.0],
    "head_state": [0.0, 0.0, 0.11464],
    "init_arm": [
        # left arm (same family as G2_STATES_WBC)
        0.5263913024,
        -0.51888638662,
        -1.0937978422,
        -1.5954054692,
        -1.1487757137,
        0.33859387489,
        -1.180540706,
        # right arm
        -0.52516957193,
        -0.51940998539,
        1.094321441,
        -1.5957545351,
        1.1482521149,
        0.33929200659,
        1.1808897719,
    ],
    "init_hand": [0.0, 0.0],
}

OPEN_THE_ROLLER_SHUTTER_G2_90D_STATES = {
    "body_state": [-0.81594142531, 1.5138985932, -0.14662238945, 0, 0],
    "head_state": [0.0, 0.0, 0.11464],
    "init_arm": [
        # left arm (same family as G2_STATES_WBC)
        0.5263913024,
        -0.51888638662,
        -1.0937978422,
        -1.5954054692,
        -1.1487757137,
        0.33859387489,
        -1.180540706,
        # right arm
        -0.52516957193,
        -0.51940998539,
        1.094321441,
        -1.5957545351,
        1.1482521149,
        0.33929200659,
        1.1808897719,
    ],
    "init_hand": [0.0, 0.0],
}

CLEAN_THE_BLACKBOARD_G2_90D_STATES = {
    "body_state": [-0.53270, 0.86830, 0.0, 0, 0],
    "head_state": [0.0, 0.0, 0.0],
    "init_arm": [
        # left arm (same family as G2_STATES_WBC)
        0.5263913024,
        -0.51888638662,
        -1.0937978422,
        -1.5954054692,
        -1.1487757137,
        0.33859387489,
        -1.180540706,
        # right arm
        -0.52516957193,
        -0.51940998539,
        1.094321441,
        -1.5957545351,
        1.1482521149,
        0.33929200659,
        1.1808897719,
    ],
    "init_hand": [0.0, 0.0],
}

TRANSPORT_CARTONS_TO_TABLE_WBC_G2_90D_STATES = {
    "body_state": [-0.53270, 0.86830, 0.0, 0, 0],
    "head_state": [0.0, 0.0, 0.0],
    "init_arm": [
        # left arm (same family as G2_STATES_WBC)
        0.5263913024,
        -0.51888638662,
        -1.0937978422,
        -1.5954054692,
        -1.1487757137,
        0.33859387489,
        -1.180540706,
        # right arm
        -0.52516957193,
        -0.51940998539,
        1.094321441,
        -1.5957545351,
        1.1482521149,
        0.33929200659,
        1.1808897719,
    ],
    "init_hand": [0.0, 0.0],
}

PUT_THE_PLATES_ON_THE_SHELF_G2_90D_STATES = {
    "body_state": [-0.53270, 0.86830, 0.0, 0, 0],
    "head_state": [0.0, 0.0, 0.11464],
    "init_arm": [
        # left arm (same family as G2_STATES_WBC)
        0.5263913024,
        -0.51888638662,
        -1.0937978422,
        -1.5954054692,
        -1.1487757137,
        0.33859387489,
        -1.180540706,
        # right arm
        -0.52516957193,
        -0.51940998539,
        1.094321441,
        -1.5957545351,
        1.1482521149,
        0.33929200659,
        1.1808897719,
    ],
    "init_hand": [0.0, 0.0],
}

CLOTH_FOLD_G2_90D_STATES = {
    "body_state": [
        -0.8039625287055969,
        1.5036951303482056,
        -0.3414643406867981,
        1.310246631192058e-07,
        2.9495149647118524e-06,
    ],
    "head_state": [-1.2831519180522832e-09, 2.0274340539749858e-10, 0.4450911283493042],
    "init_arm": [
        1.4026657342910767,
        -0.24971048533916473,
        -1.165736436843872,
        -1.686367154121399,
        -1.0325400829315186,
        -0.14668786525726318,
        -0.9705950617790222,
        -1.2026489973068237,
        -0.24966879189014435,
        1.365781545639038,
        -1.6863754987716675,
        1.2327971458435059,
        -0.14672432839870453,
        1.1706899404525757,
    ],
    "init_hand": [6.90371648737198e-11, -3.2401079597121907e-09],
}

CLOTH_FOLD_SIM_G2_90D_STATES = {
    "body_state": [
        -0.8039625287055969,
        1.5036951303482056,
        -0.3414643406867981,
        1.310246631192058e-07,
        2.9495149647118524e-06,
    ],
    "head_state": [-1.2831519180522832e-09, 2.0274340539749858e-10, 0.4450911283493042],
    "init_arm": [
        1.4026657342910767,
        -0.24971048533916473,
        -1.165736436843872,
        -1.686367154121399,
        -1.0325400829315186,
        -0.14668786525726318,
        -0.9705950617790222,
        -1.2026489973068237,
        -0.24966879189014435,
        1.365781545639038,
        -1.6863754987716675,
        1.2327971458435059,
        -0.14672432839870453,
        1.1706899404525757,
    ],
    "init_hand": [6.90371648737198e-11, -3.2401079597121907e-09],
}


G2_DEFAULT_VLN = {
    "body_state": [0.0, 0.0, 0.0, 0.0, 0.0],
    "head_state": [0.0, 0.0, 0.0],
    "init_arm": [
        -1.5720,
        1.5700,
        -1.5700,
        -1.5700,
        1.5700,
        0.0000,
        0.0000,
        1.5720,
        1.5700,
        1.5700,
        -1.5700,
        -1.5700,
        0.0000,
        0.0000,
    ],
    "init_hand": [0.0, 0.0],
}

G2_STATES_1 = {
    "body_state": [-0.83423, 1.2172, -0.18151424220741028, 0.0, 0.0],
    "head_state": [0.0, 0.0, 0.11464],
    "init_arm": [
        0.739033,
        -0.717023,
        -1.524419,
        -1.537612,
        0.27811,
        -0.925845,
        -0.839257,
        -0.739033,
        -0.717023,
        1.524419,
        -1.537612,
        -0.27811,
        -0.925845,
        0.839257,
    ],
    "init_hand": [0.0, 0.0],
}

G2_DEFAULT_2 = {
    "body_state": [-0.83423, 1.2172, 0.10025, 0.0, 0.0],
    "head_state": [0.0, 0.0, 0.0],
    "init_arm": [
        0.739033,
        -0.717023,
        -1.524419,
        -1.537612,
        0.27811,
        -0.925845,
        -0.839257,
        -0.739033,
        -0.717023,
        1.524419,
        -1.537612,
        -0.27811,
        -0.925845,
        0.839257,
    ],
    "init_hand": [0.0, 0.0],
}


G2_STATES_3 = {
    "body_state": [-0.83423, 1.2172, -0.18151424220741028, 0.0, 0.0],
    "head_state": [0.0, 0.0, 0.0],
    "init_arm": [
        0.739033,
        -0.717023,
        -1.524419,
        -1.537612,
        0.27811,
        -0.925845,
        -0.839257,
        -0.739033,
        -0.717023,
        1.524419,
        -1.537612,
        -0.27811,
        -0.925845,
        0.839257,
    ],
    "init_hand": [0.0, 0.0],
}

G2_STATES_4 = {
    "body_state": [-1.04545222194, 1.34390352404, -0.31939525311, 0.0, 1.57],
    "head_state": [0.0, 0.0, 0.11464],
    "init_arm": [
        0.739033,
        -0.717023,
        -1.524419,
        -1.537612,
        0.27811,
        -0.925845,
        -0.839257,
        -0.739033,
        -0.717023,
        1.524419,
        -1.537612,
        -0.27811,
        -0.925845,
        0.839257,
    ],
    "init_hand": [0.0, 0.0],
}

G2_STATES_5 = {
    "body_state": [-0.8342, 1.583, -0.3617, 0.0, 0.0],
    "head_state": [0.0, 0.0, 0.1745],
    "init_arm": [
        0.7459,
        -0.7458,
        -1.5375,
        -1.5828,
        0.2801,
        -0.8858,
        -0.8803,
        -0.7279,
        -0.6707,
        1.5386,
        -1.54,
        -0.283,
        -0.911,
        0.8175,
    ],
    "init_hand": [0.0, 0.0],
}

G2_STATES_6 = {
    "body_state": [-0.83423, 1.2172, -0.181514, 0.0, 0.0],
    "head_state": [0.0, 0.0, 0.0],
    "init_arm": [
        0.739033,
        -0.717023,
        -1.524419,
        -1.537612,
        0.27811,
        -0.925845,
        -0.839257,
        -0.739033,
        -0.717023,
        1.524419,
        -1.537612,
        -0.27811,
        -0.925845,
        0.839257,
    ],
    "init_hand": [0.0, 0.0],
}

G2_CRSB_DEFAULT_STATES = {
    "body_state": [-1.0, 1.638, -0.75, 0.0, 0.0],
    "head_state": [0.0, 0.0, 0.297],
    "init_arm": [
        1.73,
        -1.15,
        -1.6,
        -1.8,
        1.33,
        0.0,
        0.0,
        -1.73,
        -1.15,
        1.6,
        -1.8,
        -1.33,
        0.0,
        0.0,
    ],
    "init_hand": [0.0, 0.0],
}

# wuji five-finger hand: 40 finger joints (20/hand), init all-zeros (open).
# Arm/body/head from the wuji task pose_id 2.
G2_WUJI_DEFAULT_STATES = {
    "body_state": [-0.83423, 1.2172, 0.10025, 0.0, 0.0],
    "head_state": [0.0, 0.0, 0.11464],
    "init_arm": [
        0.739033,
        -0.717023,
        -1.524419,
        -1.537612,
        0.27811,
        -0.92,
        -0.834,
        -0.739033,
        -0.717023,
        1.524419,
        -1.537612,
        -0.27811,
        -0.92,
        0.834,
    ],
    "init_hand": [0.0] * 40,
}

# Bottle-placement init pose shared with pick_up_cola_to_coaster (G2_crsB_wuji).
# init_arm and init_hand are user-provided; body_state and head_state are
# user-provided in degrees and converted to radians.  body_state order is
# idx01..idx05 = [ankle, knee, bend, side, turn]; head_state order is
# idx11..idx13 = [turn, tilt, raise].  end_effector is metadata only
# (the actual EEF prim is determined by G2_crsB_wuji.json).
G2_WUJI_PICK_UP_BOTTLE_STATES = {
    "body_state": [
        -0.829380460548,
        1.213003830136,
        0.099483767364,
        0.0,
        0.0,
    ],
    "head_state": [
        0.0,
        0.0,
        0.113446401380,
    ],
    "init_arm": [
        0.53085124,
        -0.81865376,
        -1.6088295,
        -1.3674408,
        -0.24578807,
        -0.44023043,
        -1.0574253,
        -0.5350555,
        -0.81206584,
        1.6178627,
        -1.3703554,
        0.25676107,
        -0.44527027,
        1.0582801,
    ],
    "init_hand": [
        0.4802,
        0.0038,
        0.0092,
        0.4427,
        -0.0482,
        -0.1564,
        -0.0017,
        1.0198,
        0.0437,
        -0.2202,
        0.2089,
        0.5238,
        0.0889,
        -0.1705,
        0.0079,
        0.5597,
        -0.0078,
        -0.0834,
        -0.0059,
        0.0055,
        0.5082,
        0.1022,
        0.4864,
        0.3934,
        0.0903,
        -0.3663,
        0.5344,
        -0.0063,
        0.1888,
        -0.2774,
        0.253,
        0.0229,
        0.1692,
        -0.1805,
        0.2807,
        0.03,
        -0.0983,
        -0.124,
        0.7738,
        -0.0055,
    ],
    "end_effector": "g2a_wuji_dex",
}

G2_CRSB_STATES_1 = {
    "body_state": [-0.817169, 1.51417, -0.34577, 0, 0],
    "head_state": [0.0, 0.0, 0.34135],
    "init_arm": [
        0.5265,
        -0.5189,
        -1.0932,
        -1.5966,
        -1.148,
        0.3401,
        -1.18119,
        -0.5265,
        -0.5189,
        1.0932,
        -1.5966,
        1.148,
        0.3401,
        1.18119,
    ],
    "init_hand": [0.0, 0.0],
}

G1_STATES_1 = {
    "body_state": [0.0, 0.436, 0.349, 0.2],
    "init_arm": [
        -1.0751,
        0.6109,
        0.2793,
        -1.2846,
        0.7295,
        1.4957,
        -0.1868,
        1.0734,
        -0.6109,
        -0.2793,
        1.2846,
        -0.7313,
        -1.4957,
        0.1868,
    ],
    "init_hand": [0.0, 0.0],
}

# Bottle-placement init pose shared with pick_up_cola_to_coaster (G2_crsB_wuji).
# init_arm and init_hand are user-provided; body_state and head_state are
# user-provided in degrees and converted to radians.  body_state order is
# idx01..idx05 = [ankle, knee, bend, side, turn]; head_state order is
# idx11..idx13 = [turn, tilt, raise].  end_effector is metadata only
# (the actual EEF prim is determined by G2_crsB_wuji.json).
G2_WUJI_PICK_UP_BOTTLE_STATES = {
    "body_state": [
        -0.6981123166979405,
        1.3523379066598956,
        -0.6296161026625113,
        0.0,
        0.0,
    ],
    "head_state": [
        0.0,
        0.0,
        0.34921301919614245,
    ],
    "init_arm": [
        1.7323838256725361,
        -1.1511636983067923,
        -1.5990512941697577,
        -1.8002556445282796,
        -0.4202215732473698,
        0.004033890103143022,
        0.0007744206133841417,
        -1.7281721695666301,
        -1.1470093667433505,
        1.6010198229527115,
        -1.79933501637105,
        0.4221582239920755,
        -0.001358531735271221,
        -0.0006869357715750387,
    ],
    "init_hand": [0.0] * 40,
}

# Wuji five-finger arithmetic gesture task (wuji_arithmetic_finger_count).
# Body/head/arms use the validated GenieSim 3.2.0 wuji default pose
# (same body/head/arm as G2_DEFAULT_2) with arm joints 6/7 slightly opened
# (0.925845 -> 0.920, 0.839257 -> 0.834) per the task author. The hand pose
# is the frame-0 mean across all 500 episodes in 57916/task_28365. The captured
# pinky flexion was visibly open in preview, so it uses the adjacent ring
# finger's flexion while retaining its own lateral joint2 angle.
G2_WUJI_ARITHMETIC_STATES = {
    "body_state": [
        -0.83423,
        1.2172,
        0.10025,
        0.0,
        0.0,
    ],
    "head_state": [
        0.0,
        0.0,
        0.11464,
    ],
    # left arm joints 1-7, then right arm joints 1-7 (wrist pair tuned).
    "init_arm": [
        0.739033,
        -0.717023,
        -1.524419,
        -1.537612,
        0.27811,
        -0.920,
        -0.834,
        -0.739033,
        -0.717023,
        1.524419,
        -1.537612,
        -0.27811,
        -0.920,
        0.834,
    ],
    # 40 joints (left 20 + right 20), thumb->pinky, joint1..4.
    "init_hand": [
        1.41795588,
        -0.24997501,
        1.15711043,
        1.02390370,
        1.51451468,
        -0.11939376,
        1.24503375,
        1.29691972,
        1.42322180,
        0.01721533,
        1.43426322,
        0.77900929,
        1.52279192,
        -0.15513458,
        1.34932851,
        1.15272185,
        # Left pinky: ring flexion (joints 1/3/4), captured pinky joint2.
        1.52279192,
        0.17414281,
        1.34932851,
        1.15272185,
        1.46956350,
        0.26146126,
        1.28357493,
        1.41205982,
        1.50421582,
        0.04761484,
        1.19976340,
        1.35100825,
        1.50379122,
        -0.05150074,
        1.45251883,
        1.05032251,
        1.52581170,
        0.19126109,
        1.45979576,
        1.20816791,
        # Right pinky: ring flexion (joints 1/3/4), captured pinky joint2.
        1.52581170,
        0.13485832,
        1.45979576,
        1.20816791,
    ],
    "end_effector": "g2a_wuji_dex",
}

# Task-specific init pose for stack_bowls (G2_crsB_wuji). Body/head reuse the
# validated wuji tabletop pose; arms are a symmetric dual-arm preparation pose
# (left 7 joints first, then right 7) provided by the task author.
G2_WUJI_STACK_BOWLS_STATES = {
    "body_state": [
        -0.83423,
        1.2172,
        0.10025,
        0.0,
        0.0,
    ],
    "head_state": [
        0.0,
        0.0,
        0.11464,
    ],
    # left arm joints 1-7, then right arm joints 1-7.
    "init_arm": [
        0.739033,
        -0.717023,
        -1.524419,
        -1.537612,
        0.27811,
        -0.920,
        -0.834,
        -0.739033,
        -0.717023,
        1.524419,
        -1.537612,
        -0.27811,
        -0.920,
        0.834,
    ],
    # Open hands; the policy closes fingers around the bowls during stacking.
    # Dataset frame-0 mean max|dev| = 0.115 rad (fingers slightly curled) — kept at 0
    # like the other wuji tasks; within the hand-noise band.
    "init_hand": [0.0] * 40,
    "end_effector": "g2a_wuji_dex",
}

# Task-specific init pose for cover_cup_lid (G2_crsB_wuji). Same symmetric
# dual-arm preparation pose as stack_bowls (left 7 joints first, then right 7)
# provided by the task author; kept as a separate constant so later per-task
# tuning does not drift the validated stack_bowls pose.
G2_WUJI_COVER_CUP_LID_STATES = {
    "body_state": [
        -0.83423,
        1.2172,
        0.10025,
        0.0,
        0.0,
    ],
    "head_state": [
        0.0,
        0.0,
        0.11464,
    ],
    # left arm joints 1-7, then right arm joints 1-7.
    "init_arm": [
        0.739033,
        -0.717023,
        -1.524419,
        -1.537612,
        0.27811,
        -0.920,
        -0.834,
        -0.739033,
        -0.717023,
        1.524419,
        -1.537612,
        -0.27811,
        -0.920,
        0.834,
    ],
    # Open hands; the policy closes fingers around the lid during covering.
    "init_hand": [0.0] * 40,
    "end_effector": "g2a_wuji_dex",
}

# Task-specific init pose for take_the_scissors (G2_crsB_wuji). Same symmetric
# dual-arm preparation pose family as stack_bowls/cover_cup_lid (left 7 joints
# first, then right 7); kept as a separate constant so later per-task tuning
# does not drift the validated poses of the other tasks.
G2_WUJI_TAKE_THE_SCISSORS_STATES = {
    "body_state": [
        -0.83423,
        1.2172,
        0.10025,
        0.0,
        0.0,
    ],
    "head_state": [
        0.0,
        0.0,
        0.11464,
    ],
    # left arm joints 1-7, then right arm joints 1-7.
    "init_arm": [
        0.739033,
        -0.717023,
        -1.524419,
        -1.537612,
        0.27811,
        -0.920,
        -0.834,
        -0.739033,
        -0.717023,
        1.524419,
        -1.537612,
        -0.27811,
        -0.920,
        0.834,
    ],
    # Open hands; the policy closes fingers around the scissors during pickup.
    "init_hand": [0.0] * 40,
    "end_effector": "g2a_wuji_dex",
}

# Task-specific init pose for take_the_book (G2_crsB_wuji). Same symmetric
# dual-arm preparation pose family as take_the_scissors/cover_cup_lid (left 7
# joints first, then right 7); kept as a separate constant so later per-task
# tuning does not drift the validated poses of the other tasks. body_state
# waist pitch is adjusted (-0.225 vs 0.10025) for the higher pick plane of the
# bookshelf middle board (books at z ~0.94-1.16 vs table tops at z ~0.76).
G2_WUJI_TAKE_THE_BOOK_STATES = {
    "body_state": [
        -0.83423,
        1.2172,
        -0.225,
        0.000000,
        0.000000,
    ],
    "head_state": [
        0.000000,
        0.000000,
        0.11464,
    ],
    # left arm joints 1-7, then right arm joints 1-7.
    "init_arm": [
        0.739033,
        -0.717023,
        -1.524419,
        -1.537612,
        0.27811,
        -0.920,
        -0.834,
        -0.739033,
        -0.717023,
        1.524419,
        -1.537612,
        -0.27811,
        -0.920,
        0.834,
    ],
    "end_effector": "g2a_wuji_dex",
}

G2_CRSB_STATES_1 = {
    "body_state": [-0.817169, 1.51417, -0.34577, 0, 0],
    "head_state": [0.0, 0.0, 0.34135],
    "init_arm": [
        0.5265,
        -0.5189,
        -1.0932,
        -1.5966,
        -1.148,
        0.3401,
        -1.18119,
        -0.5265,
        -0.5189,
        1.0932,
        -1.5966,
        1.148,
        0.3401,
        1.18119,
    ],
    "init_hand": [0.0, 0.0],
}

G1_STATES_1 = {
    "body_state": [0.0, 0.436, 0.349, 0.2],
    "init_arm": [
        -1.0751,
        0.6109,
        0.2793,
        -1.2846,
        0.7295,
        1.4957,
        -0.1868,
        1.0734,
        -0.6109,
        -0.2793,
        1.2846,
        -0.7313,
        -1.4957,
        0.1868,
    ],
    "init_hand": [0.0, 0.0],
}

ARX_DEFAULT_STATES = {
    "body_state": [],
    "head_state": [],
    "init_arm": [
        -0.0002,
        0.9588,
        0.8536,
        -0.5804,
        -0.0002,
        0.0044,
        -0.0002,
        0.9588,
        0.8536,
        -0.5804,
        -0.0002,
        0.0044,
    ],
    "init_hand": [0.044, 0.044],
}

ARX_REAL_STATES = {
    "body_state": [],
    "head_state": [],
    "init_arm": [
        # 左臂 6 关节（实机采集）
        -0.0006, 0.9482, 0.8562, -0.5770, -0.0006, -0.0162,
        # 右臂 6 关节（实机采集）
        0.0002, 0.9470, 0.8566, -0.5770, -0.0032, -0.0158,
    ],
    "init_hand": [0.044, 0.044],
}

AGILEX_DEFAULT_STATES = {
    "body_state": [],
    "head_state": [],
    "init_arm": [
        -0.18,
        1.35,
        -0.84,
        0.0,
        0.0,
        0.0,
        0.18,
        1.35,
        -0.84,
        0.0,
        0.0,
        0.0,
    ],
    "init_hand": [0.04, 0.04],
}

LIFT2_DEFAULT_STATES = {
    "body_state": [0.46],
    "head_state": [],
    "init_arm": [0.0] * 12,
    "init_hand": [0.044, 0.044],
}

SPLIT_ALOHA_DEFAULT_STATES = {
    "body_state": [-0.64],
    "head_state": [],
    "init_arm": [0.0] * 12,
    "init_hand": [0.05, 0.05],
}

INTERN_LIFT2_ZERO_STATES = {
    **LIFT2_DEFAULT_STATES,
    "init_arm": [0.0] * 12,
}

INTERN_SPLIT_ALOHA_ZERO_STATES = {
    **SPLIT_ALOHA_DEFAULT_STATES,
    "init_arm": [0.0] * 12,
}

# Per-task split_aloha arm init; same 6-joint pose applied to both arms.
INTERN_SPLIT_ALOHA_MICROWAVE_STATES = {
    **SPLIT_ALOHA_DEFAULT_STATES,
    "init_arm": [4.8e-3, 0.342, -0.140, 1.7e-2, 0.144, -2.5e-3] * 2,
}

INTERN_SPLIT_ALOHA_HANG_CUP_STATES = {
    **SPLIT_ALOHA_DEFAULT_STATES,
    "init_arm": [4.8e-3, 0.342, -0.140, 1.7e-2, 0.144, -2.5e-3] * 2,
}

G2_STATES_TABLEPICK = {
    "body_state": [-0.80192, 1.50468, -0.34060, 0.0, 0.0],
    "head_state": [0.0, 0.0, 0.348733],
    "init_arm": [
        0.52474,
        -0.51835,
        -1.09429,
        -1.59523,
        -1.14905,
        0.33841,
        -1.18067,
        -0.52494,
        -0.52010,
        1.09500,
        -1.59592,
        1.14757,
        0.33804,
        1.18074,
    ],
    "init_hand": [0.0, 0.0],
}

G2_STATES_TIGHTEN = {
    "body_state": [-0.55470, 1.16102, -0.17399, 0.0, 0.0],
    "head_state": [0.0, 0.0, 0.24280],
    "init_arm": [
        1.1755601846837085,
        -0.6027659661118391,
        -1.6180251984089145,
        -1.877490139091333,
        -0.9228874233086497,
        0.19786099129442564,
        -1.3104225024956555,
        -1.1632921333851767,
        -0.6184447278184354,
        1.61470820463961,
        -1.8253683481329563,
        0.8885854557310436,
        0.1705115519534133,
        1.3418106256297733,
    ],
    "init_hand": [0.0, 0.0],
}

# Mean across the 10 plug_charging_cable_into_phone instances
# (instruction_<uuid>.json.robot_state_frame0.joints). The capture is essentially
# identical across instances (max spread 1.6e-4 rad on init_arm, 2e-6 on the
# waist), so one canonical init pose covers all ten. body_state matches
# G2_DEFAULT_STATES; head_state pitches further down (0.30 vs 0.11464) to look
# at the low table this task uses.
G2_STATES_PHONE_CHARGE = {
    "body_state": [-0.83422, 1.21724, 0.10027, 0.0, 0.0],
    "head_state": [0.0, 0.0, 0.30000],
    "init_arm": [
        1.302998,
        -0.349669,
        -1.265729,
        -1.786361,
        -1.132125,
        -0.246669,
        -1.070621,
        -1.302935,
        -0.349669,
        1.265704,
        -1.786368,
        1.132190,
        -0.246668,
        1.070623,
    ],
    "init_hand": [0.0, 0.0],
}

# Episode-0 initial poses for the crsB 90d fine-manipulation tasks.
# Keep each task's waist, head and both arms together with its eval-task base
# pose; the charging-cable pose does not reproduce these recorded viewpoints.
# Values are rounded to 1e-6 rad; nominal-zero joint noise below 1e-5 is omitted.
G2_STATES_BUILDING_BLOCK_REPLAY_0 = {

    "body_state": [-0.754289, 1.545668, -0.341840, 0.0, 0.0],
    "head_state": [0.0, -0.000996, 0.449825],
    "init_arm": [
        0.525800, -0.480989, -1.119135, -1.574436, -1.190575, 0.402583, -1.186754,
        -0.445167, -0.503239, 1.116656, -1.625538, 1.138010, 0.294651, 1.196086,
    ],
    "init_hand": [0.0, 0.0],
}

G2_STATES_BEARING_REPLAY_0 = {

    "body_state": [-0.829849, 1.435976, -0.341842, 0.0, 0.0],
    "head_state": [0.0, -0.000999, 0.449834],
    "init_arm": [
        0.521977, -0.496605, -1.084397, -1.569143, -1.142833, 0.375997, -1.141371,
        -0.540090, -0.525881, 1.104042, -1.525223, 1.168638, 0.254565, 1.233745,
    ],
    "init_hand": [0.0, 0.0],
}

# Gearing-package startup pose.
G2_STATES_BEARING_GEARING_20261020 = {
    "body_state": [-0.8029, 1.5048, -0.3419, 0, 0],
    "head_state": [0, -0.001, 0.45],
    "init_arm": [
        0.5264, -0.5189, -1.0938, -1.5954, -1.1488, 0.3386, -1.1805,
        -0.5252, -0.5194, 1.0943, -1.5958, 1.1483, 0.3393, 1.1809,
    ],
    "init_hand": [0, 0],
}

G2_STATES_ETHERNET_REPLAY_0 = {

    "body_state": [-0.834215, 1.217235, 0.100266, 0.0, 0.0],
    "head_state": [0.0, 0.0, 0.200000],
    "init_arm": [
        1.303042, -0.349669, -1.265725, -1.786360, -1.132103, -0.246674, -1.070622,
        -1.302980, -0.349669, 1.265700, -1.786368, 1.132170, -0.246673, 1.070621,
    ],
    "init_hand": [0.0, 0.0],
}

G2_STATES_STRAW_REPLAY_0 = {


    "body_state": [-0.802876, 1.504990, -0.341443, 0.0, 0.0],
    "head_state": [0.0, 0.0, 0.445030],
    "init_arm": [
        1.304262, -0.349645, -1.265581, -1.786343, -1.131352, -0.246805, -1.070621,
        -1.304171, -0.349649, 1.265567, -1.786337, 1.131456, -0.246802, 1.070657,
    ],
    "init_hand": [0.0, 0.0],
}

# Pen-cap and coin tasks use separate initial waist and arm poses.
G2_STATES_PEN_CAP_REPLAY_0 = {
    "body_state": [-0.851189, 1.493458, -0.341443, 0.0, 0.0],
    "head_state": [0.0, 0.0, 0.445030],
    "init_arm": [
        1.315224, -0.396690, -1.329740, -1.842499, -1.145075, -0.278220, -1.065289,
        -1.312697, -0.356178, 1.356087, -1.822663, 1.156954, -0.239925, 1.041300,
    ],
    "init_hand": [0.0, 0.0],
}

G2_STATES_COIN_REPLAY_0 = {
    "body_state": [-0.792790, 1.553110, -0.341442, 0.0, 0.0],
    "head_state": [0.0, 0.0, 0.445029],
    "init_arm": [
        1.293886, -0.320660, -1.248959, -1.728574, -1.151529, -0.264311, -1.098752,
        -1.286044, -0.387926, 1.297764, -1.851113, 1.158769, -0.228678, 1.043324,
    ],
    "init_hand": [0.0, 0.0],
}

# Task-specific init pose for pick_up_a_package (G2_crsB_wuji). Reuses the
# pick-up-package asset pose (pose_id 1). init_hand is all-zeros (open);
# the policy closes the fingers around the box during the bimanual side grasp.
G2_WUJI_DEFAULT_STATES_id1 = {
    "body_state": [
        -0.6981123166979405,
        1.3523379066598956,
        -0.6296161026625113,
        0.0,
        0.0,
    ],
    "head_state": [
        0.0,
        0.0,
        0.34921301919614245,
    ],
    "init_arm": [
        1.7323838256725361,
        -1.1511636983067923,
        -1.5990512941697577,
        -1.8002556445282796,
        -0.4202215732473698,
        0.004033890103143022,
        0.0007744206133841417,
        -1.7281721695666301,
        -1.1470093667433505,
        1.6010198229527115,
        -1.79933501637105,
        0.4221582239920755,
        -0.001358531735271221,
        -0.0006869357715750387,
    ],
    "init_hand": [0.0] * 40,
}

# Task-specific init pose for stack_bowls (G2_crsB_wuji). Body/head reuse the
# validated wuji tabletop pose; arms are a symmetric dual-arm preparation pose
# (left 7 joints first, then right 7) provided by the task author.
G2_WUJI_STACK_BOWLS_STATES = {
    "body_state": [
        -0.83423,
        1.2172,
        0.10025,
        0.0,
        0.0,
    ],
    "head_state": [
        0.0,
        0.0,
        0.11464,
    ],
    # left arm joints 1-7, then right arm joints 1-7.
    "init_arm": [
        0.739033,
        -0.717023,
        -1.524419,
        -1.537612,
        0.27811,
        -0.920,
        -0.834,
        -0.739033,
        -0.717023,
        1.524419,
        -1.537612,
        -0.27811,
        -0.920,
        0.834,
    ],
    # Open hands; the policy closes fingers around the bowls during stacking.
    # Dataset frame-0 mean max|dev| = 0.115 rad (fingers slightly curled) — kept at 0
    # like the other wuji tasks; within the hand-noise band.
    "init_hand": [0.0] * 40,
    "end_effector": "g2a_wuji_dex",
}

# Task-specific init pose for cover_cup_lid (G2_crsB_wuji). Same symmetric
# dual-arm preparation pose as stack_bowls (left 7 joints first, then right 7)
# provided by the task author; kept as a separate constant so later per-task
# tuning does not drift the validated stack_bowls pose.
G2_WUJI_COVER_CUP_LID_STATES = {
    "body_state": [
        -0.83423,
        1.2172,
        0.10025,
        0.0,
        0.0,
    ],
    "head_state": [
        0.0,
        0.0,
        0.11464,
    ],
    # left arm joints 1-7, then right arm joints 1-7.
    "init_arm": [
        0.739033,
        -0.717023,
        -1.524419,
        -1.537612,
        0.27811,
        -0.920,
        -0.834,
        -0.739033,
        -0.717023,
        1.524419,
        -1.537612,
        -0.27811,
        -0.920,
        0.834,
    ],
    # Open hands; the policy closes fingers around the lid during covering.
    "init_hand": [0.0] * 40,
    "end_effector": "g2a_wuji_dex",
}

# Task-specific init pose for take_the_scissors (G2_crsB_wuji). Same symmetric
# dual-arm preparation pose family as stack_bowls/cover_cup_lid (left 7 joints
# first, then right 7); kept as a separate constant so later per-task tuning
# does not drift the validated poses of the other tasks.
G2_WUJI_TAKE_THE_SCISSORS_STATES = {
    "body_state": [
        -0.83423,
        1.2172,
        0.10025,
        0.0,
        0.0,
    ],
    "head_state": [
        0.0,
        0.0,
        0.11464,
    ],
    # left arm joints 1-7, then right arm joints 1-7.
    "init_arm": [
        0.739033,
        -0.717023,
        -1.524419,
        -1.537612,
        0.27811,
        -0.920,
        -0.834,
        -0.739033,
        -0.717023,
        1.524419,
        -1.537612,
        -0.27811,
        -0.920,
        0.834,
    ],
    # Open hands; the policy closes fingers around the scissors during pickup.
    "init_hand": [0.0] * 40,
    "end_effector": "g2a_wuji_dex",
}

# Task-specific init pose for take_the_book (G2_crsB_wuji). Same symmetric
# dual-arm preparation pose family as take_the_scissors/cover_cup_lid (left 7
# joints first, then right 7); kept as a separate constant so later per-task
# tuning does not drift the validated poses of the other tasks. body_state
# waist pitch is adjusted (-0.225 vs 0.10025) for the higher pick plane of the
# bookshelf middle board (books at z ~0.94-1.16 vs table tops at z ~0.76).
G2_WUJI_TAKE_THE_BOOK_STATES = {
    "body_state": [
        -0.83423,
        1.2172,
        -0.225,
        0.000000,
        0.000000,
    ],
    "head_state": [
        0.000000,
        0.000000,
        0.11464,
    ],
    # left arm joints 1-7, then right arm joints 1-7.
    "init_arm": [
        0.739033,
        -0.717023,
        -1.524419,
        -1.537612,
        0.27811,
        -0.920,
        -0.834,
        -0.739033,
        -0.717023,
        1.524419,
        -1.537612,
        -0.27811,
        -0.920,
        0.834,
    ],
    # Open hands; the policy closes fingers around the book during pickup.
    "init_hand": [0.0] * 40,
    "end_effector": "g2a_wuji_dex",
}

TASK_INFO_DICT = {
    "phone_packaging": {
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
    },
    "plug_charging_cable_into_phone": {
        "G2_crsB_90d": G2_STATES_PHONE_CHARGE,
    },
    "insert_building_block": {
        "G2_crsB_90d": G2_STATES_BUILDING_BLOCK_REPLAY_0,
    },
    "insert_ethernet_cable": {
        "G2_crsB_90d": G2_STATES_ETHERNET_REPLAY_0,
    },
    "insert_straw_into_milk_tea_cup": {
        "G2_crsB_90d": G2_STATES_STRAW_REPLAY_0,
    },
    "install_bearing_shaft_seat5mm": {
        "G2_crsB_90d": G2_STATES_BEARING_GEARING_20261020,
    },
    "arrange_block": {
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_largest_number": {
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_object_color": {
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_object_common_sense": {
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_billiards_color": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "arx": ARX_REAL_STATES,
        "agilex": AGILEX_DEFAULT_STATES,
    },
    "open_short_trash_can_and_throw_garbage_multi_robot": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
        "arx": ARX_DEFAULT_STATES,
        "agilex": AGILEX_DEFAULT_STATES,
    },
    "hang_cups_on_cup_tree_arxone": {
        "arx": ARX_DEFAULT_STATES,
    },
    "hang_cups_on_cup_tree_agilex": {
        "agilex": AGILEX_DEFAULT_STATES,
    },
    "hang_cups_on_cup_tree_g1": {
        "G1_omnipicker": G1_DEFAULT_STATES,
    },
    "hang_cups_on_cup_tree_g2_omnipicker": {
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "scrape_chicken_bone_into_bin_arxone": {
        "arx": ARX_DEFAULT_STATES,
    },
    "scrape_chicken_bone_into_bin_agilex": {
        "agilex": AGILEX_DEFAULT_STATES,
    },
    "scrape_chicken_bone_into_bin_g1": {
        "G1_omnipicker": G1_DEFAULT_STATES,
    },
    "scrape_chicken_bone_into_bin_g2_omnipicker": {
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "keyboard_keycap_swap_arxone": {
        "arx": ARX_DEFAULT_STATES,
    },
    "keyboard_keycap_swap_agilex": {
        "agilex": AGILEX_DEFAULT_STATES,
    },
    "keyboard_keycap_swap_g1": {
        "G1_omnipicker": G1_DEFAULT_STATES,
    },
    "keyboard_keycap_swap_g2_omnipicker": {
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "write_uppercase_letter_with_pen_arxone": {
        "arx": ARX_DEFAULT_STATES,
    },
    "write_uppercase_letter_with_pen_agilex": {
        "agilex": AGILEX_DEFAULT_STATES,
    },
    "write_uppercase_letter_with_pen_g1": {
        "G1_omnipicker": G1_DEFAULT_STATES,
    },
    "write_uppercase_letter_with_pen_g2_omnipicker": {
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    # Platform defaults, like every other multi-robot task. The delivered
    # package records a different right-arm pose in its task_info.json, but its
    # own docs say the robot and task initialization layers come from the
    # existing runtime environment (STARTUP.md), so that record is a snapshot of
    # the sender's environment rather than a contract to reproduce. Keeping the
    # defaults also makes the starting pose identical across all five
    # multi-robot tasks, so results stay comparable between them.
    "pour_50ml_into_graduated_cylinder_arxone": {
        "arx": ARX_DEFAULT_STATES,
    },
    "pour_50ml_into_graduated_cylinder_agilex": {
        "agilex": AGILEX_DEFAULT_STATES,
    },
    "pour_50ml_into_graduated_cylinder_g1": {
        "G1_omnipicker": G1_DEFAULT_STATES,
    },
    "pour_50ml_into_graduated_cylinder_g2_omnipicker": {
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "wuji_pick_block": {
        "G2_crsB_wuji": G2_WUJI_DEFAULT_STATES,
    },
    "pick_up_cola_to_coaster": {
        "G2_crsB_wuji": G2_WUJI_PICK_UP_BOTTLE_STATES,
    },
    "pick_block_color": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
        "G2_90d_gp": G2_DEFAULT_STATES,
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_omnipicker": G2_CRSB_DEFAULT_STATES,
        "G2_crsB_wuji": G2_WUJI_DEFAULT_STATES,
        "arx": ARX_REAL_STATES,
        "agilex": AGILEX_DEFAULT_STATES,
    },
    "ci_pick_block_color": {
        "G2_crsB_omnipicker": G2_CRSB_DEFAULT_STATES,
    },
    "pick_block_color_instructgen": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
        "G2_90d": G2_90D_DEFAULT_STATES,
    },
    "pick_block_shape": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "arx": ARX_REAL_STATES,
        "agilex": AGILEX_DEFAULT_STATES,
    },
    "pick_block_size": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "arx": ARX_REAL_STATES,
        "agilex": AGILEX_DEFAULT_STATES,
    },
    "pick_block_number": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "arx": ARX_REAL_STATES,
        "agilex": AGILEX_DEFAULT_STATES,
    },
    "pick_cup_size": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_fruit_size": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_object_type": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "arx": ARX_REAL_STATES,
        "agilex": AGILEX_DEFAULT_STATES,
    },
    "pick_pen_color": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_specific_object": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "arx": ARX_REAL_STATES,
        "agilex": AGILEX_DEFAULT_STATES,
    },
    "pick_specific_object_unseen": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "arx": ARX_REAL_STATES,
        "agilex": AGILEX_DEFAULT_STATES,
    },
    "pick_accessory": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_object_position": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_stationery": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_follow_logic_and": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_follow_logic_not": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_follow_logic_or": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "arx": ARX_REAL_STATES,
        "agilex": AGILEX_DEFAULT_STATES,
    },
    "place_block_into_box": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "stable_grasp": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "place_object_into_box_position": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "place_object_into_box_color": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "place_object_into_box_size": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "put_pen_into_penholder": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "sort_accessory": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "straighten_object": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "arx": ARX_REAL_STATES,
        "agilex": AGILEX_DEFAULT_STATES,
    },
    "sort_fruit": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "sort_fruit_unseen": {
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pack_groceries_into_carton": {
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "place_eggs_into_holder": {
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "set_the_table": {
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_object_relative_position_unseen": {
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "place_book": {"G2_omnipicker": G2_STATES_1},
    "place_book_hard": {"G2_omnipicker": G2_STATES_1},
    "open_door": {"G2_omnipicker": G2_STATES_1},
    "bimanual_hold_ball": {"G2_omnipicker": G2_STATES_1},
    "take_book": {"G2_omnipicker": G2_STATES_1},
    "clean_the_desktop": {"G2_omnipicker": G2_DEFAULT_STATES},
    "goal_image_clean_the_desktop": {"G2_omnipicker": G2_DEFAULT_STATES},
    "clean_the_desktop_unseen": {"G2_omnipicker": G2_DEFAULT_STATES},
    "pick_object_absolute_position_unseen": {"G2_omnipicker": G2_DEFAULT_STATES},
    "empty_desktop_bin": {"G2_omnipicker": G2_DEFAULT_STATES},
    "dump_trash_kitchen": {"G2_omnipicker": G2_STATES_1},
    "throw_away_garbage": {"G2_omnipicker": G2_DEFAULT_STATES},
    "heat_food": {"G2_omnipicker": G2_STATES_1},
    "hold_pot": {"G2_omnipicker": G2_STATES_1},
    "hang_tableware": {"G2_omnipicker": G2_STATES_1},
    "put_utensil_turn_faucet": {"G2_omnipicker": G2_STATES_1},
    "put_pen_into_penholder": {"G2_omnipicker": G2_STATES_1},
    "store_objects_in_drawer": {"G2_omnipicker": G2_STATES_3},
    "pick_common_sense": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "arx": ARX_REAL_STATES,
        "agilex": AGILEX_DEFAULT_STATES,
    },
    "pour_workpiece": {"G2_omnipicker": G2_STATES_1},
    "scoop_popcorn": {"G2_omnipicker": G2_STATES_6},
    "sorting_packages": {"G2_omnipicker": G2_STATES_4},
    "sorting_packages_continuous": {"G2_omnipicker": G2_STATES_4},
    "chassis_at_target": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "take_wrong_item_shelf": {
        "G2_omnipicker": G2_STATES_5,
        "G1_omnipicker": G1_DEFAULT_STATES,
    },
    "stock_and_straighten_shelf": {
        "G2_omnipicker": G2_STATES_1,
    },
    "goal_image_stock_and_straighten_shelf": {
        "G2_omnipicker": G2_STATES_1,
    },
    "bimanual_chip_handover": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
    },
    "pack_in_supermarket": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_PACK_STATES,
    },
    "stack_blocks": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "place_block_into_drawer": {
        "G1_omnipicker": G1_STATES_1,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "sort_number": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "sort_beverage": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "stack_cubes": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "stack_bowls": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
        "G2_crsB_wuji": G2_WUJI_STACK_BOWLS_STATES,
        "G2_crsB_wuji": G2_WUJI_DEFAULT_STATES,
    },
    "find_block_in_three_storage_boxes": {
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "stack_three_building_blocks": {
        "G2_omnipicker": G2_DEFAULT_STATES,
        "arx": ARX_REAL_STATES,
    },
    "sort_cubes_by_size": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_object_absolute_position": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_object_relative_position": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "place_object_relative_position": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "goal_image_place_object_relative_position": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "place_beverage_to_anothers_position": {
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "goal_image_place_beverage_to_anothers_position": {
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "vln_test": {
        "G2_omnipicker": G2_DEFAULT_VLN,
    },
    "wbc_pick_object": {
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
    },
    "wbc_pick_trash": {
        "G2_crsB_90d": G2_CRSB_STATES_1,
    },
    "wbc_pick_object_and_place": {
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
    },
    "wbc_pick_object_from_shelf_high": {
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
    },
    "wbc_pick_object_from_shelf_low": {
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
    },
    "find_target_in_drawer": {
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "stack_multiple_blocks": {
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "stack_blocks_based_on_pattern": {
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    # InternDataEngine scene pack (intern_*)
    "intern_close_the_microwave": {
        "G2_omnipicker": G2_DEFAULT_STATES,
        "arx": ARX_REAL_STATES,
        "agilex": AGILEX_DEFAULT_STATES,
        "lift2": INTERN_LIFT2_ZERO_STATES,
        "split_aloha": INTERN_SPLIT_ALOHA_MICROWAVE_STATES,
    },
    "intern_heat_the_food_in_the_microwave": {
        "G2_omnipicker": G2_DEFAULT_STATES,
        "arx": ARX_REAL_STATES,
        "agilex": AGILEX_DEFAULT_STATES,
        "lift2": INTERN_LIFT2_ZERO_STATES,
    },
    "intern_pull_the_storagefurniture": {
        "G2_omnipicker": G2_DEFAULT_STATES,
        "arx": ARX_REAL_STATES,
        "agilex": AGILEX_DEFAULT_STATES,
        "lift2": INTERN_LIFT2_ZERO_STATES,
        "split_aloha": INTERN_SPLIT_ALOHA_ZERO_STATES,
    },
    "intern_hang_the_cup_on_rack": {
        "G2_omnipicker": G2_DEFAULT_STATES,
        "arx": ARX_REAL_STATES,
        "agilex": AGILEX_DEFAULT_STATES,
        "lift2": INTERN_LIFT2_ZERO_STATES,
        "split_aloha": INTERN_SPLIT_ALOHA_HANG_CUP_STATES,
    },
    "intern_insert_the_flower_in_vase": {
        "G2_omnipicker": G2_DEFAULT_STATES,
        "arx": ARX_REAL_STATES,
        "agilex": AGILEX_DEFAULT_STATES,
        "lift2": INTERN_LIFT2_ZERO_STATES,
        "split_aloha": INTERN_SPLIT_ALOHA_ZERO_STATES,
    },
    "intern_insert_the_markpen_in_penholder": {
        "G2_omnipicker": G2_DEFAULT_STATES,
        "arx": ARX_REAL_STATES,
        "agilex": AGILEX_DEFAULT_STATES,
        "lift2": INTERN_LIFT2_ZERO_STATES,
        "split_aloha": INTERN_SPLIT_ALOHA_ZERO_STATES,
    },
    "intern_pick_the_package_on_conveyor": {
        "G2_omnipicker": G2_DEFAULT_STATES,
        "arx": ARX_REAL_STATES,
        "agilex": AGILEX_DEFAULT_STATES,
        "lift2": INTERN_LIFT2_ZERO_STATES,
    },
    "intern_stack_bowls": {
        "G2_omnipicker": G2_DEFAULT_STATES,
        "arx": ARX_REAL_STATES,
        "agilex": AGILEX_DEFAULT_STATES,
        "lift2": INTERN_LIFT2_ZERO_STATES,
        "split_aloha": INTERN_SPLIT_ALOHA_ZERO_STATES,
    },
    "intern_pack_in_objects": {
        "G2_omnipicker": G2_DEFAULT_STATES,
        "arx": ARX_REAL_STATES,
        "agilex": AGILEX_DEFAULT_STATES,
        "lift2": INTERN_LIFT2_ZERO_STATES,
    },
    "intern_pack_out_objects": {
        "G2_omnipicker": G2_DEFAULT_STATES,
        "arx": ARX_REAL_STATES,
        "agilex": AGILEX_DEFAULT_STATES,
        "lift2": INTERN_LIFT2_ZERO_STATES,
        "split_aloha": INTERN_SPLIT_ALOHA_ZERO_STATES,
    },
    "intern_stack_a_beef_sandwich": {
        "G2_omnipicker": G2_DEFAULT_STATES,
        "arx": ARX_REAL_STATES,
        "agilex": AGILEX_DEFAULT_STATES,
        "lift2": INTERN_LIFT2_ZERO_STATES,
    },
    "intern_store_objects_in_the_drawer": {
        "G2_omnipicker": G2_DEFAULT_STATES,
        "arx": ARX_REAL_STATES,
        "agilex": AGILEX_DEFAULT_STATES,
        "lift2": INTERN_LIFT2_ZERO_STATES,
    },
    # ---- g290d room2 generalization tasks ----
    "pick_single_object": {
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_specific_object_new": {
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_object_of_specific_category": {
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_building_block_of_specific_color": {
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_building_block_of_specific_size": {
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_object_of_specific_size": {
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_object_of_specific_shape": {
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_object_in_specific_container": {
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_home_tableware": {
        "G2_90d": G2_90D_HOME_00_STATES,
    },
    "pick_building_block_of_specific_position": {
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_building_block_of_specific_corner_position": {
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_building_block_of_relative_position": {
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_combined_grasp_of_multiple_targets": {
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pick_combined_grasp_with_arm_and_position": {
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "place_object_into_container": {
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "place_object_into_named_container": {
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "place_object_into_container_of_specific_color": {
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "place_object_into_sized_container": {
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "place_object_into_container_of_specific_shape": {
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "place_object_into_positional_container": {
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "place_object_into_corner_container": {
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "place_object_at_relative_position": {
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "place_combined_object_into_container": {
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pass_multiple_object": {
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "pass_single_object": {
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_90D_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
    },
    "insert_coin_into_money_box": {
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_STATES_COIN_REPLAY_0,
    },
    "put_the_cap_on_the_pen_new": {
        "G2_90d": G2_90D_DEFAULT_STATES,
        "G2_crsB_90d": G2_STATES_PEN_CAP_REPLAY_0,
    },
    # "<sub_task>@<background>" beats the plain sub_task key (base_env), so the
    # g2op_s2r suite gets the real robot's rest pose without touching the same
    # sub_tasks under the if / robust / zeroshot backgrounds.
    "bimanual_chip_handover@table_task_g2_op_s2r": {
        "G2_omnipicker": G2_S2R_STATES,
    },
    "pick_specific_object@table_task_g2_op_s2r": {
        "G2_omnipicker": G2_S2R_STATES,
    },
    "place_object_into_box_color@table_task_g2_op_s2r": {
        "G2_omnipicker": G2_S2R_STATES,
    },
    "pack_in_supermarket@table_task_g2_op_s2r": {
        "G2_omnipicker": G2_S2R_STATES,
    },
    "pick_block_color@table_task_g2_op_s2r": {
        "G2_omnipicker": G2_S2R_STATES,
    },
    "pick_block_size@table_task_g2_op_s2r": {
        "G2_omnipicker": G2_S2R_STATES,
    },
    "sort_fruit@table_task_g2_op_s2r": {
        "G2_omnipicker": G2_S2R_STATES,
    },
    "place_block_into_drawer@drawer_task_g2_op_s2r": {
        "G2_omnipicker": G2_S2R_STATES,
    },
    "stack_three_building_blocks_large_arx": {
        "arx": ARX_REAL_STATES,
    },
    "pick_up_a_package": {
        "G2_crsB_wuji": G2_WUJI_DEFAULT_STATES_id1,
    },
    "cover_cup_lid": {
        "G2_crsB_wuji": G2_WUJI_COVER_CUP_LID_STATES,
    },
    "take_the_scissors": {
        "G2_crsB_wuji": G2_WUJI_TAKE_THE_SCISSORS_STATES,
    },
    "take_the_book": {
        "G2_crsB_wuji": G2_WUJI_TAKE_THE_BOOK_STATES,
    },
    "open_the_door": {
        "G2_90d": OPEN_THE_DOOR_G2_90D_STATES,
        "G2_crsB_90d": OPEN_THE_DOOR_G2_90D_STATES,
    },
    "sending_food": {
        "G2_90d": SENDING_FOOD_G2_90D_STATES,
        "G2_crsB_90d": SENDING_FOOD_G2_90D_STATES,
    },
    "pick_up_trash_wbc": {
        "G2_90d": G2_STATES_WBC,
        "G2_crsB_90d": G2_STATES_WBC,
    },
    "match_and_pick_from_conveyor": {
        "G2_90d": MATCH_AND_PICK_FROM_CONVEYOR_G2_90D_STATES,
        "G2_crsB_90d": MATCH_AND_PICK_FROM_CONVEYOR_G2_90D_STATES,
    },
    "put_the_plates_on_the_shelf": {
        "G2_90d": PUT_THE_PLATES_ON_THE_SHELF_G2_90D_STATES,
        "G2_crsB_90d": PUT_THE_PLATES_ON_THE_SHELF_G2_90D_STATES,
    },
    "open_the_roller_shutter": {
        "G2_90d": OPEN_THE_ROLLER_SHUTTER_G2_90D_STATES,
        "G2_crsB_90d": OPEN_THE_ROLLER_SHUTTER_G2_90D_STATES,
    },
    "clean_the_blackboard": {
        "G2_90d": CLEAN_THE_BLACKBOARD_G2_90D_STATES,
        "G2_crsB_90d": CLEAN_THE_BLACKBOARD_G2_90D_STATES,
    },
    "transport_cartons_to_table_wbc": {
        "G2_90d": TRANSPORT_CARTONS_TO_TABLE_WBC_G2_90D_STATES,
        "G2_crsB_90d": TRANSPORT_CARTONS_TO_TABLE_WBC_G2_90D_STATES,
    },
    # New instructions and geometry use separate foreground identities.
    "stack_bowls_wuji": {
        "G1_omnipicker": G1_DEFAULT_STATES,
        "G1_120s": G1_DEFAULT_STATES,
        "G2_omnipicker": G2_DEFAULT_STATES,
        "G2_crsB_wuji": G2_WUJI_STACK_BOWLS_STATES,
        "G2_crsB_wuji": G2_WUJI_DEFAULT_STATES,
    },
}


# Named joint values are taken from frame zero of each source recording.
TASK_INFO_DICT.update(
    {name: {"G2_crsB_90d": task["robot_init_state"]} for name, task in SANDBOX_TASKS.items()}
)

"""Constants for replay and recording."""

# Asynchronous write limits.
SAVE_WORKERS = 4
MAX_INFLIGHT_SAVE_TASKS = 64

# G1/G2 recording settings mirror the task recording_setting entries.
G1_RECORDING_SETTING = {
    "camera_list": [
        "/G1/head_link2/Head_Camera",
        "/G1/gripper_r_base_link/Right_Camera",
        "/G1/gripper_l_base_link/Left_Camera",
    ],
    "fps": 30,
    "noised_probability": 0.0,
}
G2_RECORDING_SETTING = {
    "camera_list": [
        "/G2/head_link3/head_front_Camera",
        "/G2/gripper_r_base_link/Right_Camera",
        "/G2/gripper_l_base_link/Left_Camera",
        "/G2/head_link3/head_right_Camera",
        "/G2/head_link3/head_left_Camera",
    ],
    "fps": 30,
    "noised_probability": 0.0,
}
ARX_RECORDING_SETTING = {
    "camera_list": [
        "/ARX/base_link/head_camera",
        "/ARX/arm_l_end_link/hand_left_camera",
        "/ARX/arm_r_end_link/hand_right_camera",
    ],
    "fps": 30,
    "noised_probability": 0.0,
}

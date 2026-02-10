int32 GROUP_HEAD_YAW=0
int32 GROUP_HEAD_PITCH=1
int32 GROUP_LEFT_ARM=2
int32 GROUP_RIGHT_ARM=3
int32 GROUP_LEFT_TOOL=4
int32 GROUP_RIGHT_TOOL=5
int32 GROUP_WAIST_LIFT=6
int32 GROUP_WAIST_PITCH=7
int32 GROUP_CHASSIS=8
int32 GROUP_HEAD_ROLL=9
int32 GROUP_WAIST=10
int32 group_id
int32 control_type
string frame_id

string[] target_frame_names
geometry_msgs/Pose[] target_frame_poses
geometry_msgs/Pose[] target_frame_poses_delta
geometry_msgs/Twist[] target_frame_velocities

string[] reference_frame_names
geometry_msgs/Pose[] reference_frame_poses

float64[] target_joint_positions
float64[] target_joint_positions_delta
float64[] target_joint_velocities

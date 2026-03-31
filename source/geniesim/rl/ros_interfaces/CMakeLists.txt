cmake_minimum_required(VERSION 3.8)
project(geniesim_rl_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetObjectPose.srv"
  "srv/GetObjectAABB.srv"
  "srv/ResetScene.srv"
)

ament_package()

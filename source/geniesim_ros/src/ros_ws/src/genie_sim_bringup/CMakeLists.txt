cmake_minimum_required(VERSION 3.10)
project(genie_sim_bringup)

find_package(genie_sim_common REQUIRED)
genie_sim_package()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

install(
  DIRECTORY launch rviz config
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/gripper_cmds.py
    scripts/wbc_cmds.py
    scripts/wok_flip_cmds.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(
    assemble_pipeline_test
    test/test_assemble_pipeline.py
  )
  ament_add_pytest_test(wbc_cmds_test test/test_wbc_cmds.py)
  ament_add_pytest_test(
    physics_arguments_test
    test/test_physics_arguments.py
  )
endif()

ament_package()

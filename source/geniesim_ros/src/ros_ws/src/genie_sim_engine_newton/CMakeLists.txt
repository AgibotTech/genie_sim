cmake_minimum_required(VERSION 3.16)
project(genie_sim_engine_newton NONE)

find_package(genie_sim_common REQUIRED)
genie_sim_package()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Python library modules — installed as a proper ament_python package so
# that ``import genie_sim_engine_newton.*`` works from any process.
# ament_python_install_package walks genie_sim_engine_newton/ and installs
# it under the workspace's python<X>/site-packages so it's importable
# workspace-wide.
ament_python_install_package(${PROJECT_NAME})

install(
  PROGRAMS
    scripts/genie_sim_engine_newton_node.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(gl_observation_tests tests/test_gl_observation.py TIMEOUT 30)
  ament_add_pytest_test(gl_video_tests tests/test_gl_video.py TIMEOUT 30)
  ament_add_pytest_test(session_topology_tests tests/test_session_topology.py TIMEOUT 30)
  ament_add_pytest_test(isaacsim_renderer_tests tests/test_isaacsim_renderer.py TIMEOUT 60)
  ament_add_pytest_test(scene_snapshot_tests tests/test_scene_snapshot.py TIMEOUT 30)
  ament_add_pytest_test(wok_evaluation_tests tests/test_wok_evaluation.py TIMEOUT 30)
  ament_add_pytest_test(wok_contacts_tests tests/test_wok_contacts.py TIMEOUT 30)
  ament_add_pytest_test(wok_evaluation_ros_tests tests/test_wok_evaluation_ros.py TIMEOUT 30)
endif()

ament_package()

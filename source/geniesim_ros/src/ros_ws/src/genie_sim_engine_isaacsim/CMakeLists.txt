cmake_minimum_required(VERSION 3.16)
project(genie_sim_engine_isaacsim NONE)

find_package(genie_sim_common REQUIRED)
genie_sim_package()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Python library modules — installed as a proper ament_python package so
# that ``import genie_sim_engine_isaacsim.kit.bootstrap`` and friends
# work workspace-wide. ament_python_install_package walks
# genie_sim_engine_isaacsim/ and installs it under the workspace's
# python<X>/site-packages.
ament_python_install_package(${PROJECT_NAME})

install(
  PROGRAMS
    scripts/genie_sim_engine_isaacsim_node.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(isaacsim_contract_tests tests)
endif()

ament_package()

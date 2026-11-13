cmake_minimum_required(VERSION 3.16)
project(genie_sim_engine_libuipc NONE)
find_package(genie_sim_common REQUIRED)
genie_sim_package()
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS
  scripts/genie_sim_engine_libuipc_node.py
  examples/dump_fold_cloth_particles.py
  test/native_smoke.py
  DESTINATION lib/${PROJECT_NAME})
install(FILES examples/README.md DESTINATION share/${PROJECT_NAME}/examples)
install(FILES genie_sim_engine_libuipc/solver/LICENSE genie_sim_engine_libuipc/solver/PROVENANCE.md
  DESTINATION share/${PROJECT_NAME}/solver)
install(FILES README.md AGENTS.md DESIGN.md LICENSE DESTINATION share/${PROJECT_NAME})
if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(episode_sequence_tests test TIMEOUT 60
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR})
endif()
ament_package()

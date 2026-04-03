cmake_minimum_required(VERSION 3.8)
project(parol6_launch)

find_package(ament_cmake REQUIRED)

# Install launch files, controller config, and scene configs
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

# Install Python scripts as executables
install(
  PROGRAMS scripts/scene_marker_publisher.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME scene_marker_publisher)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

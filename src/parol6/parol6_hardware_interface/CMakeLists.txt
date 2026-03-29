cmake_minimum_required(VERSION 3.8)
project(parol6_hardware_interface)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── Dependencies ──────────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(pluginlib REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

# MuJoCo via the vendor package (built during docker image build)
find_package(mujoco_vendor REQUIRED)

# abseil
find_package(absl REQUIRED)

# ── Shared library ────────────────────────────────────────────────────────────
add_library(${PROJECT_NAME} SHARED
  src/parol6_hardware_interface.cpp
  src/parol6_backend.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../../common/backend/mujoco_backend.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# common/backend headers live under src/
target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# MuJoCo headers (mark as SYSTEM to suppress warnings)
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC
  ${mujoco_vendor_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
  yaml-cpp
  mujoco::mujoco
  absl::synchronization
)

ament_target_dependencies(${PROJECT_NAME}
  hardware_interface
  rclcpp
  rclcpp_lifecycle
  pluginlib
  sensor_msgs
)

# ── pluginlib export ──────────────────────────────────────────────────────────
pluginlib_export_plugin_description_file(hardware_interface parol6_hardware_interface_plugin.xml)

# ── Install ───────────────────────────────────────────────────────────────────
install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(
  hardware_interface
  rclcpp
  rclcpp_lifecycle
  pluginlib
  sensor_msgs
)

ament_package()

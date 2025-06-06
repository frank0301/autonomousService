cmake_minimum_required(VERSION 3.5)
project(common_interface)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
# Uncomment and add other dependencies as needed
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # The following line skips the linter which checks for copyrights
  # Comment the line when a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# Generate messages and services
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/KeyCtrl.msg"
  "msg/SerComStruct.msg"
  "msg/RectDepth.msg"
  "msg/Camera2map.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# Install package.xml
install(
  FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

# Export dependencies
ament_export_dependencies(rosidl_default_runtime)
ament_package()

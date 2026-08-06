cmake_minimum_required(VERSION 3.8)
project(flight_manager)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(custom_msgs REQUIRED)

add_executable(flight_manager src/flight_manager.cpp)
ament_target_dependencies(flight_manager
  rclcpp
  std_msgs
  px4_msgs
  custom_msgs
)

install(TARGETS flight_manager
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

cmake_minimum_required(VERSION 3.8)
project(flight_driver)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(custom_msgs REQUIRED)

add_executable(flight_driver src/flight_driver.cpp)
ament_target_dependencies(flight_driver rclcpp std_msgs geometry_msgs px4_msgs custom_msgs)

install(TARGETS flight_driver DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY missions DESTINATION share/${PROJECT_NAME})

ament_package()

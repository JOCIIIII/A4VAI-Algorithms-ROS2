cmake_minimum_required(VERSION 3.8)
project(rs_converter)

# Default to C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

# Add executables
add_executable(rs_converter src/rs_converter.cpp)
ament_target_dependencies(rs_converter
  rclcpp
  sensor_msgs
  pcl_conversions
  PCL
  Eigen3
)

add_executable(point_cloud_filter_node src/point_cloud_filter_node.cpp)
ament_target_dependencies(point_cloud_filter_node
  rclcpp
  sensor_msgs
  pcl_conversions
  PCL
  nav_msgs
  tf2_ros
  std_msgs
  tf2
  tf2_geometry_msgs
  Eigen3
)


add_executable(point_cloud_feature_extractor src/point_cloud_feature_extractor.cpp)
add_executable(point_cloud_feature_extractor_cov src/point_cloud_feature_extractor_cov.cpp)

ament_target_dependencies(point_cloud_feature_extractor rclcpp pcl_conversions sensor_msgs nav_msgs geometry_msgs)
ament_target_dependencies(point_cloud_feature_extractor_cov rclcpp pcl_conversions sensor_msgs nav_msgs geometry_msgs)

# Link PCL libraries
target_link_libraries(rs_converter ${PCL_LIBRARIES})
target_link_libraries(point_cloud_filter_node ${PCL_LIBRARIES})
target_link_libraries(point_cloud_feature_extractor ${PCL_LIBRARIES})
target_link_libraries(point_cloud_feature_extractor_cov ${PCL_LIBRARIES})


# Install executables
install(TARGETS
  rs_converter
  point_cloud_filter_node
  point_cloud_feature_extractor
  point_cloud_feature_extractor_cov
  DESTINATION lib/${PROJECT_NAME}
)

# Call ament_package
ament_package()


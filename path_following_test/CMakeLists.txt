cmake_minimum_required(VERSION 3.8)
project(path_following_test)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(custom_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)

add_executable(path_following_test src/path_following_test.cpp)
ament_target_dependencies(path_following_test
  rclcpp
  std_msgs
  px4_msgs
  custom_msgs
  ament_index_cpp
)

install(TARGETS path_following_test
  DESTINATION lib/${PROJECT_NAME}
)

# wp.csv 를 패키지 share 에 설치 (노드가 ament share 경로에서 기본 로드)
install(FILES wp.csv
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

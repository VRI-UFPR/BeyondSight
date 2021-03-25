cmake_minimum_required(VERSION 2.8.3)
project(laser_to_pointcloud2)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  sensor_msgs
  tf
  laser_geometry
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(laser_to_pointcloud2 src/laser_to_pointcloud2.cpp)
target_link_libraries(laser_to_pointcloud2 ${catkin_LIBRARIES})
add_dependencies(laser_to_pointcloud2 ${catkin_EXPORTED_TARGETS})

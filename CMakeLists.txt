cmake_minimum_required(VERSION 3.0.2)
project(kopi)


## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  std_msgs
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  CATKIN_DEPENDS geometry_msgs roscpp rospy std_msgs
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Shutdown Node
add_executable(shutdown_node src/poweroff.cpp)
target_link_libraries(shutdown_node ${catkin_LIBRARIES})
add_dependencies(shutdown_node ${catkin_EXPORTED_TARGETS})

#############
## Install ##
#############

install(TARGETS shutdown_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

cmake_minimum_required(VERSION 2.8.3)
project(ackermann_drive)

# Check for c++11 / c++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "-std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "-std=c++0x")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

find_package(catkin REQUIRED COMPONENTS
    # OpenCV
    cv_bridge
    rospy
    geometry_msgs
    sensor_msgs
    nav_msgs
    roscpp
    gazebo_ros
)

find_package(gazebo REQUIRED)

link_directories(${GAZEBO_LIBRARY_DIRS})
include_directories(${Boost_INCLUDE_DIR} ${catkin_INCLUDE_DIRS} ${GAZEBO_INCLUDE_DIRS})

catkin_package(
    # INCLUDE_DIRS include
    # LIBRARIES 
    # CATKIN_DEPENDS rospy 
    CATKIN_DEPENDS rospy roscpp std_msgs message_runtime gazebo_ros
)

# target_link_libraries ( my_exec ${OpenCV_LIBRARIES} ${catkin_LIBRARIES} )

include_directories(
    # ${OpenCV_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
)

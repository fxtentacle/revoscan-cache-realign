cmake_minimum_required(VERSION 3.25)
project(cache_realign)

set(CMAKE_CXX_STANDARD 17)

add_executable(cache_realign main.cpp)

find_package(PCL 1.3 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(nlohmann_json 3.2.0 REQUIRED)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

target_link_libraries(cache_realign ${PCL_LIBRARIES} nlohmann_json::nlohmann_json Eigen3::Eigen)

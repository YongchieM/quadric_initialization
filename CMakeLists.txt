cmake_minimum_required(VERSION 2.9)
project(myppf)

include_directories("/usr/include/eigen3/")

find_package(PCL 1.8 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(THIRD_PARTY_LIBS
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES})
    
set(SOURCES
    src/test.cpp
    src/t_hash_int.cpp
    src/pose_3d.cpp
    src/ppf_helpers.cpp
    src/ppf_match.cpp)
    
add_executable(test ${SOURCES})
target_link_libraries(test PRIVATE ${THIRD_PARTY_LIBS} ${PROJECT_SOURCE_DIR}/include)
install(TARGETS test RUNTIME DESTINATION bin)

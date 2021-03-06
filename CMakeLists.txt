cmake_minimum_required(VERSION 2.9)
project(LfD)


include_directories("/usr/local/include/eigen3/")

#list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Pangolin)


include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(${Pangolin_INCLUDE_DIRS})

set(THIRD_PARTY_LIBS
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${Pangolin_LIBRARIES})


set(SOURCES
    src/test.cpp
    src/estimate_ellipsoids.cpp
    src/viewer.cpp)
    
add_executable(test ${SOURCES})
target_link_libraries(test PRIVATE ${THIRD_PARTY_LIBS} ${PROJECT_SOURCE_DIR}/include)
install(TARGETS test RUNTIME DESTINATION bin)

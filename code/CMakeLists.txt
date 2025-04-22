##############################################
# CMakeLists for GPU Project
##############################################
cmake_minimum_required(VERSION 3.5)
if(NOT OpenCV_DIR)
    message(FATAL_ERROR " Must set OpenCV_DIR: cmake -DOpenCV_DIR=path/to/opencv/build")
else()
### benchmark
    project(benchmark)
    find_package(OpenCV REQUIRED)
    add_executable(${PROJECT_NAME} benchmark.cpp benchmark.hh)
    target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})
endif()
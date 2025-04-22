##############################################
# CMakeLists for GPU Project
##############################################
cmake_minimum_required(VERSION 3.5)
if(NOT OpenCV_DIR)
    message(FATAL_ERROR " Must set OpenCV_DIR: cmake -DOpenCV_DIR=path/to/opencv/build")
else()
### Canny
    project(canny)
    find_package(OpenCV REQUIRED)
    add_executable(canny canny.cpp)
    target_include_directories(canny PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(canny PUBLIC ${OpenCV_LIBS})
endif()
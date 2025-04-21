##############################################
# CMakeLists for GPU Project
##############################################
cmake_minimum_required(VERSION 3.5)
if(NOT OpenCV_DIR)
    message(FATAL_ERROR " Must set OpenCV_DIR: cmake -DOpenCV_DIR=path/to/opencv/install")
else()
    find_package(CUDAToolkit)
    find_package(OpenCV 4.2.0 REQUIRED)
    ### Canny
    project(canny)
    add_executable(canny canny.cpp)
    target_include_directories(canny PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
    target_link_libraries(canny PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES})
    message(STATUS "LIBS: ${OpenCV_LIBS}")
endif()
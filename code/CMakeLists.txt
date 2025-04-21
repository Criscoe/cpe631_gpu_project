##############################################
# CMakeLists for GPU Project
##############################################
cmake_minimum_required(VERSION 3.5)
set(OpenCV_DIR "C:\\Users\\Caleb\\Documents\\GitHub\\OpenCvGPU\\build2\\install")
find_package(OpenCV 4.2.0 REQUIRED)
find_package(CUDA REQUIRED)
### Canny
project(canny)
add_executable(canny canny.cpp)
target_include_directories(canny PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
target_link_libraries(canny PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES})
message(STATUS "LIBS: ${OpenCV_LIBS}")
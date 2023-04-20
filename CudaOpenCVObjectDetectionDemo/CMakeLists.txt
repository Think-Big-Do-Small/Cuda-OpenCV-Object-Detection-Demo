cmake_minimum_required(VERSION 3.0.0)

find_package(OpenCV REQUIRED)
MESSAGE(STATUS "Include dirs ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "LINK LIBRARIES ${OpenCV_LIBS}")

project(CudaOpenCVObjectDetection VERSION 0.1.0)





include(CTest)
enable_testing()

add_executable(CudaOpenCVObjectDetection main.cpp)


target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

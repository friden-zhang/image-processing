cmake_minimum_required(VERSION 3.10)

project(color-convert)

set(CMAKE_CXX_STANDARD 17)

find_package(TBB REQUIRED)

include_directories(${TBB_INCLUDE_DIRS})

file(GLOB CPU_SOURCES "kernels/cpu/*.cc")

file(GLOB CUDA_SOURCES "kernels/cuda/*.cu")

add_library(${PROJECT_NAME}-cpu SHARED ${CPU_SOURCES})
target_link_libraries(${PROJECT_NAME}-cpu PUBLIC TBB::tbb)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    target_compile_definitions(${PROJECT_NAME}-cpu PRIVATE __ARM_NEON__)
endif()

file(GLOB CPU_SOURCES "kernels/*.cc")
add_library(${PROJECT_NAME} SHARED ${CPU_SOURCES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_NAME}-cpu)

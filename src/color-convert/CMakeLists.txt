cmake_minimum_required(VERSION 3.10)

project(color-convert LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(TBB REQUIRED)
# 使用 QUIET 参数查找 CUDA，如果找不到则不会报错
find_package(CUDA)

include_directories(${TBB_INCLUDE_DIRS})

# 设置宏 NO_CUDA 和 CUDA_INCLUDE_DIRS
if (CUDA_FOUND)
    enable_language(CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
    set(HAS_CUDA TRUE)
else()
    message(WARNING "CUDA not found. Defining NO_CUDA macro.")
    add_compile_definitions(NO_CUDA)
    set(HAS_CUDA FALSE)
endif()

file(GLOB CPU_SOURCES "kernels/cpu/*.cc")

# 创建 CPU 版本的库
add_library(${PROJECT_NAME}-cpu SHARED ${CPU_SOURCES})
target_link_libraries(${PROJECT_NAME}-cpu PUBLIC TBB::tbb)

# 如果找到 CUDA，创建 CUDA 版本的库
if (HAS_CUDA)
    file(GLOB CUDA_SOURCES "kernels/cuda/*.cu")
    set(CMAKE_CUDA_ARCHITECTURES "60;61;70;75;80")
    add_library(${PROJECT_NAME}-cuda SHARED ${CUDA_SOURCES})
    target_link_libraries(${PROJECT_NAME}-cuda PUBLIC ${CUDA_LIBRARIES})
endif()

# 检查是否为 ARM 架构
if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    target_compile_definitions(${PROJECT_NAME}-cpu PRIVATE __ARM_NEON__)
endif()

# 检查是否为 x86 架构
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|i[3-6]86")
    target_compile_definitions(${PROJECT_NAME}-cpu PRIVATE __AVX__)
    target_compile_options(${PROJECT_NAME}-cpu PRIVATE -mavx)
    target_compile_options(${PROJECT_NAME}-cpu PRIVATE -mavx2)
endif()

# 创建主库，并根据 CUDA 的可用性链接相应的库
file(GLOB MAIN_SOURCES "kernels/*.cc")
add_library(${PROJECT_NAME} SHARED ${MAIN_SOURCES})

if (HAS_CUDA)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_NAME}-cpu ${PROJECT_NAME}-cuda)
else()
    target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_NAME}-cpu)
endif()

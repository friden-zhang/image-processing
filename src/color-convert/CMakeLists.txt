
find_package(TBB REQUIRED)

include_directories(${TBB_INCLUDE_DIRS})


file(GLOB CPU_SOURCES "kernels/cpu/*.cc")

add_library(color-convert-cpu SHARED ${CPU_SOURCES})
target_link_libraries(color-convert-cpu PUBLIC TBB::tbb)

if (HAS_CUDA)
    file(GLOB CUDA_SOURCES "kernels/cuda/*.cu")
    set(CMAKE_CUDA_ARCHITECTURES "60;61;70;75;80")
    add_library(color-convert-cuda SHARED ${CUDA_SOURCES})
    target_link_libraries(color-convert-cuda PUBLIC ${CUDA_LIBRARIES})
endif()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
    target_compile_definitions(color-convert-cpu PRIVATE __ARM_NEON__)
endif()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|i[3-6]86")
    target_compile_definitions(color-convert-cpu PRIVATE __AVX__)
    target_compile_options(color-convert-cpu PRIVATE -mavx)
    target_compile_options(color-convert-cpu PRIVATE -mavx2)
endif()

file(GLOB MAIN_SOURCES "kernels/*.cc")
add_library(color-convert SHARED ${MAIN_SOURCES})

if (HAS_CUDA)
    target_link_libraries(color-convert PUBLIC color-convert-cpu color-convert-cuda)
else()
    target_link_libraries(color-convert PUBLIC color-convert-cpu)
endif()

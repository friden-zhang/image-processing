cmake_minimum_required(VERSION 3.10)
project(ImageProcessing LANGUAGES CXX CUDA)

include_directories(include)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

# Find OpenMP
find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(CUDA)

if (CUDA_FOUND)
    enable_language(CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
    set(CMAKE_CUDA_ARCHITECTURES "50;60;70;75;80;89")
    add_compile_definitions(HAS_CUDA)
    set(HAS_CUDA TRUE)
else()
    message(WARNING "CUDA not found. Defining NO_CUDA macro.")
    add_compile_definitions(NO_CUDA)
    set(HAS_CUDA FALSE)
endif()

# set realease build type
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# -O2
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")

add_subdirectory(src/color-convert)

add_subdirectory(tests)

add_subdirectory(benchmarks)
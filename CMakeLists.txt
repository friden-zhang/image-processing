cmake_minimum_required(VERSION 3.10)
project(ImageProcessing)

include_directories(include)

# -O2
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")

add_subdirectory(src/color-convert)

add_subdirectory(tests)

add_subdirectory(benchmarks)
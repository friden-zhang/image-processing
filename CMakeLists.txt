cmake_minimum_required(VERSION 3.10)
project(ImageProcessing)

include_directories(include)


add_subdirectory(src/color-convert)

add_subdirectory(tests)

add_subdirectory(benchmarks)
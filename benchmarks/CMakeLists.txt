cmake_minimum_required(VERSION 3.10)

project(image-processing-benchmark)

set(CMAKE_CXX_STANDARD 17)

find_package(benchmark REQUIRED)

include_directories(${benchmark_INCLUDE_DIRS})

aux_source_directory(./ SOURCE_FILES)

add_executable(run_benchmarks ${SOURCE_FILES})
target_link_libraries(run_benchmarks benchmark::benchmark color-convert)

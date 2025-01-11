
find_package(benchmark REQUIRED)

include_directories(${benchmark_INCLUDE_DIRS})

aux_source_directory(./ SOURCE_FILES)

add_executable(run_benchmarks ${SOURCE_FILES})
target_link_libraries(run_benchmarks benchmark::benchmark color-convert)

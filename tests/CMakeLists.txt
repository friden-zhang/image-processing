cmake_minimum_required(VERSION 3.10)
project(image-processing-tests)

find_package(GTest REQUIRED)
find_package(PNG REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${PNG_INCLUDE_DIRS})

aux_source_directory(./ SOURCE_FILES)

add_executable(run_tests ${SOURCE_FILES})

target_link_libraries(run_tests ${GTEST_LIBRARIES} pthread PNG::PNG color-convert)
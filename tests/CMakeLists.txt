
find_package(GTest REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${PNG_INCLUDE_DIRS})

aux_source_directory(./ SOURCE_FILES)

add_executable(run_tests ${SOURCE_FILES})

target_link_libraries(run_tests ${GTEST_LIBRARIES} pthread color-convert)
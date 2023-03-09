cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(tak_cpp)

include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

file(GLOB all_include
        "${PROJECT_SOURCE_DIR}/include/*.h"
        )

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(train_env "${PROJECT_SOURCE_DIR}/src/train_env.cpp" "${PROJECT_SOURCE_DIR}/src/board.cpp" "${PROJECT_SOURCE_DIR}/src/util.cpp"  ${all_include})
target_link_libraries(train_env "${TORCH_LIBRARIES}")
set_property(TARGET train_env PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
# file(GLOB TORCH_DLLS "C:/Users/Jesse/Projects/tak_cpp/libtorch_release/lib/*.dll")
file(GLOB TORCH_DLLS "C:/Users/Jesse/Projects/tak_cpp/lib/libtorch_debug/lib/*.dll")
add_custom_command(TARGET train_env
                    POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${TORCH_DLLS}
                    $<TARGET_FILE_DIR:train_env>)


add_executable(test_env "${PROJECT_SOURCE_DIR}/src/test_env.cpp" "${PROJECT_SOURCE_DIR}/src/board.cpp" "${PROJECT_SOURCE_DIR}/src/util.cpp"  ${all_include})
target_link_libraries(test_env "${TORCH_LIBRARIES}")
set_property(TARGET test_env PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
# file(GLOB TORCH_DLLS "C:/Users/Jesse/Projects/tak_cpp/libtorch_release/lib/*.dll")
file(GLOB TORCH_DLLS "C:/Users/Jesse/Projects/tak_cpp/lib/libtorch_debug/lib/*.dll")
add_custom_command(TARGET test_env
                    POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${TORCH_DLLS}
                    $<TARGET_FILE_DIR:test_env>)
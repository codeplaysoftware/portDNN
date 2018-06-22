# Copyright 2018 Codeplay Software Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use these files except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


cmake_minimum_required(VERSION 3.2.2)
project(snn_bench)
include(HandleBenchmark)
include(SNNHelpers)

# Creates a git-config.tmp file containing the commit hash and date. A custom
# target ensures that this command is always executed.
add_custom_target(generate_git_config
  ${CMAKE_CURRENT_SOURCE_DIR}/make_git_config.sh >
    ${CMAKE_CURRENT_BINARY_DIR}/git_config.tmp
  BYPRODUCTS
    git_config.tmp
  WORKING_DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Conditionally copies the new git-config.tmp to git-config.h if the file has
# changed. This pattern is used to prevent constant rebuilds even though the git
# history hasn't changed.
add_custom_command(OUTPUT git_config.h
  COMMAND
    ${CMAKE_COMMAND} -E copy_if_different git_config.tmp git_config.h
  DEPENDS
    generate_git_config git_config.tmp
  WORKING_DIRECTORY
    ${CMAKE_CURRENT_BINARY_DIR}
)
add_library(bench_main STATIC bench_main.cc git_config.h)
target_link_libraries(bench_main benchmark::benchmark)
target_include_directories(bench_main PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

snn_bench(
  WITH_SYCL
  TARGET
    set_buffer
  SOURCES
    basic_sycl.cc
  PUBLIC_LIBRARIES
    bench_main
)
snn_bench(
  WITH_SYCL
  TARGET
    simple_convolution
  SOURCES
    simple_convolution.cc
  OBJECTS
    $<TARGET_OBJECTS:direct_conv2d>
    $<TARGET_OBJECTS:tiled_conv2d>
  PUBLIC_LIBRARIES
    bench_main
    Eigen::Eigen
)

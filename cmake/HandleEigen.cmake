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

if(NOT SNN_DOWNLOAD_EIGEN)
  find_package(Eigen)
endif()

if(NOT Eigen_FOUND AND (SNN_DOWNLOAD_EIGEN OR SNN_DOWNLOAD_MISSING_DEPS))
  message(STATUS "Configuring Eigen library")
  set(EIGEN_HG_TAG "ComputeCpp-v0.6.0" CACHE STRING
    "Hg tag, branch or commit to use for the Eigen library"
  )
  configure_file(${CMAKE_SOURCE_DIR}/cmake/EigenDownload.cmake.in
    ${sycldnn_BINARY_DIR}/eigen-download/CMakeLists.txt)

  execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${sycldnn_BINARY_DIR}/eigen-download
  )
  if(result)
    message(FATAL_ERROR "CMake step for Eigen failed: ${result}")
  endif()

  execute_process(COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${sycldnn_BINARY_DIR}/eigen-download
  )
  if(result)
    message(FATAL_ERROR "Download step for Eigen failed: ${result}")
  endif()

  find_package(Eigen)
endif()

if(NOT Eigen_FOUND)
  message(FATAL_ERROR
    "Could not find Eigen, consider setting SNN_DOWNLOAD_MISSING_DEPS")
endif()

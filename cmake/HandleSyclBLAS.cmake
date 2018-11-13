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

if(NOT SNN_DOWNLOAD_SYCLBLAS)
  find_package(SyclBLAS)
endif()

if(NOT SyclBLAS_FOUND AND (SNN_DOWNLOAD_SYCLBLAS OR SNN_DOWNLOAD_MISSING_DEPS))
  message(STATUS "Configuring SyclBLAS library")
  # Select a commit from the SyclBLAS master branch. This should be manually
  # bumped as appropriate.
  set(SyclBLAS_GIT_TAG "15335a5" CACHE STRING
    "Git tag, branch or commit to use for the SyclBLAS library"
  )
  configure_file(${CMAKE_SOURCE_DIR}/cmake/SyclBLASDownload.cmake.in
    ${sycldnn_BINARY_DIR}/sycl-blas-download/CMakeLists.txt)

  execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${sycldnn_BINARY_DIR}/sycl-blas-download
  )
  if(result)
    message(FATAL_ERROR "CMake step for SyclBLAS failed: ${result}")
  endif()

  execute_process(COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${sycldnn_BINARY_DIR}/sycl-blas-download
  )
  if(result)
    message(FATAL_ERROR "Download step for SyclBLAS failed: ${result}")
  endif()

  find_package(SyclBLAS)
endif()

if(NOT SyclBLAS_FOUND)
  message(FATAL_ERROR
    "Could not find SyclBLAS, consider setting SNN_DOWNLOAD_MISSING_DEPS")
endif()

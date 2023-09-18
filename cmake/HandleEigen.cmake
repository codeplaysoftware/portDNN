# Copyright Codeplay Software Ltd.
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

cmake_minimum_required(VERSION 3.10.2)

include(SNNHelpers)
snn_include_guard(HANDLE_EIGEN)

if(NOT SNN_DOWNLOAD_EIGEN)
  find_package(Eigen)
endif()

if(NOT Eigen_FOUND AND (SNN_DOWNLOAD_EIGEN OR SNN_DOWNLOAD_MISSING_DEPS))
  include(ExternalProject)
  set(EIGEN_REPO "https://gitlab.com/libeigen/eigen.git" CACHE STRING
    "Eigen repository to use"
  )
  set(EIGEN_GIT_TAG "3.4.0" CACHE STRING
    "Git tag, branch or commit to use for the Eigen library"
  )
  set(EIGEN_SOURCE_DIR ${portdnn_BINARY_DIR}/Eigen-src)
  if(NOT TARGET Eigen_download)
    ExternalProject_Add(Eigen_download
      GIT_REPOSITORY    ${EIGEN_REPO}
      GIT_TAG           ${EIGEN_GIT_TAG}
      GIT_SHALLOW       ON
      GIT_CONFIG        advice.detachedHead=false
      SOURCE_DIR        ${EIGEN_SOURCE_DIR}
      CONFIGURE_COMMAND ""
      BUILD_COMMAND     ""
      INSTALL_COMMAND   ""
      TEST_COMMAND      ""
    )
  endif()
  set(EIGEN_INCLUDE_DIR ${EIGEN_SOURCE_DIR})
  file(MAKE_DIRECTORY ${EIGEN_INCLUDE_DIR})

  find_package(Eigen)
  add_dependencies(Eigen::Eigen Eigen_download)
  mark_as_advanced(EIGEN_REPO EIGEN_GIT_TAG)
endif()

if(NOT Eigen_FOUND)
  message(FATAL_ERROR
    "Could not find Eigen, consider setting SNN_DOWNLOAD_MISSING_DEPS")
endif()

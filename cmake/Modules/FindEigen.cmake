# Copyright 2018 Codeplay Software Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use these files except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Try to find the Eigen library and its Tensor module.
#
# If the library is found then the `eigen::eigen` target will be exported with
# the required include directories.
#
# Sets the following variables:
#   eigen_FOUND        - whether the system has Eigen
#   eigen_INCLUDE_DIRS - the Eigen include directory
#
find_path(EIGEN_INCLUDE_DIR
  NAMES unsupported/Eigen/CXX11/Tensor
  PATH_SUFFIXES eigen3 Eigen3
  DOC "The Eigen SYCL Tensor module"
)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Eigen
  FOUND_VAR EIGEN_FOUND
  REQUIRED_VARS EIGEN_INCLUDE_DIR
)
mark_as_advanced(EIGEN_FOUND EIGEN_INCLUDE_DIRS)
if(EIGEN_FOUND)
  set(EIGEN_INCLUDE_DIRS ${EIGEN_INCLUDE_DIR})
endif()

if(EIGEN_FOUND AND NOT TARGET Eigen::Eigen)
  add_library(Eigen::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${EIGEN_INCLUDE_DIR}"
  )
  set_target_properties(Eigen::Eigen PROPERTIES
    INTERFACE_COMPILE_DEFINITIONS DISABLE_SKINNY=1
  )
endif()


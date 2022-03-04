# Copyright Codeplay Software Ltd.
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

find_path(EIGEN_INCLUDE_DIR
  NAMES unsupported/Eigen/CXX11/Tensor
  PATH_SUFFIXES eigen3 Eigen3
  DOC "The Eigen SYCL Tensor module"
)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Eigen
  FOUND_VAR Eigen_FOUND
  REQUIRED_VARS EIGEN_INCLUDE_DIR
)
mark_as_advanced(Eigen_FOUND EIGEN_INCLUDE_DIRS)
if(Eigen_FOUND)
  set(EIGEN_INCLUDE_DIRS ${EIGEN_INCLUDE_DIR})
endif()

if(Eigen_FOUND AND NOT TARGET Eigen::Eigen)
  add_library(Eigen::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${EIGEN_INCLUDE_DIR}"
  )
endif()
if(Eigen_FOUND)
  set(eigen_definitions EIGEN_SYCL_DISABLE_SKINNY=1
                        EIGEN_SYCL_DISABLE_RANK1=1
                        EIGEN_SYCL_DISABLE_GEMV=1
                        EIGEN_SYCL_DISABLE_SCALAR=1
                        EIGEN_HAS_CXX11_MATH=1
                        EIGEN_EXCEPTIONS
                        EIGEN_USE_SYCL)
  find_package(Threads)
  if(Threads_FOUND)
    list(APPEND eigen_definitions EIGEN_SYCL_ASYNC_EXECUTION=1)
    set_property(TARGET Eigen::Eigen
      APPEND PROPERTY INTERFACE_LINK_LIBRARIES Threads::Threads
    )
  endif()
  if(SNN_EIGEN_NO_BARRIER)
    list(APPEND eigen_definitions EIGEN_SYCL_DISABLE_ARM_GPU_CACHE_OPTIMISATION=1
                                  EIGEN_SYCL_NO_LOCAL_MEM=1)
  else()
    if(SNN_EIGEN_LOCAL_MEM)
      list(APPEND eigen_definitions EIGEN_SYCL_LOCAL_MEM=1)
    endif()
    if(SNN_EIGEN_NO_LOCAL_MEM)
      list(APPEND eigen_definitions EIGEN_SYCL_NO_LOCAL_MEM=1)
    endif()
  endif()
  set_target_properties(Eigen::Eigen PROPERTIES
    INTERFACE_COMPILE_DEFINITIONS "${eigen_definitions}"
  )
  if(SNN_EIGEN_COMPRESS_NAMES)
    set_target_properties(Eigen::Eigen PROPERTIES
      INTERFACE_COMPUTECPP_FLAGS "-sycl-compress-name"
    )
  endif()
endif()

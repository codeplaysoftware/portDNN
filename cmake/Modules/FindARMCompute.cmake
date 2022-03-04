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


# Try to find the ARMCompute library.
#
# If the library is found then the ARMCompute::ARMCompute, ARMCompute::Core and 
# ARMCompute::Graph targets will be exported with the required include 
# directories.
#
# Sets the following variables:
#   ARMCompute_FOUND - whether the system has ARMCompute
#   ARMCompute_INCLUDE_DIRS - the ARMCompute include directory

if(DEFINED ARM_COMPUTE_ROOT_DIR)
  message(DEPRECATION
    "ARM_COMPUTE_ROOT_DIR is deprecated, use ARMCompute_DIR instead.")
  set(ARMCompute_DIR ${ARM_COMPUTE_ROOT_DIR})
endif()

find_path(ARMCompute_INCLUDE_DIR
  NAMES arm_compute/graph.h
  PATHS ${ARMCompute_DIR}
  DOC "ARM Compute Library public headers"
)
find_path(ARMCompute_DEPENDS_INCLUDE_DIR
  NAMES half/half.hpp
  PATHS ${ARMCompute_DIR}/include
  DOC "ARM Compute Library dependency headers"
)
find_library(ARMCompute_LIBRARY arm_compute ${ARMCompute_DIR}/build)
find_library(ARMComputeCore_LIBRARY arm_compute_core ${ARMCompute_DIR}/build)
find_library(ARMComputeGraph_LIBRARY arm_compute_graph ${ARMCompute_DIR}/build)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(ARMCompute
  FOUND_VAR
    ARMCompute_FOUND
  REQUIRED_VARS 
    ARMCompute_INCLUDE_DIR
    ARMCompute_LIBRARY
    ARMComputeCore_LIBRARY
    ARMComputeGraph_LIBRARY
)

mark_as_advanced(ARMCompute_FOUND ARMCompute_INCLUDE_DIR)

if(ARMCompute_FOUND)
  set(ARMCompute_INCLUDE_DIRS
    ${ARMCompute_INCLUDE_DIR}
    ${ARMCompute_DEPENDS_INCLUDE_DIR}
  )
endif()

if(ARMCompute_FOUND AND NOT TARGET ARMCompute::ARMCompute)
  add_library(ARMCompute::ARMCompute IMPORTED SHARED)
  set_target_properties(ARMCompute::ARMCompute PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES
      "${ARMCompute_INCLUDE_DIRS}"
    INTERFACE_COMPILE_DEFINITIONS
      ARM_COMPUTE
    IMPORTED_LOCATION
      ${ARMCompute_LIBRARY}
    IMPORTED_NO_SONAME
      TRUE
  )

  add_library(ARMCompute::Core IMPORTED SHARED)
  set_target_properties(ARMCompute::Core PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES
      "${ARMCompute_INCLUDE_DIRS}"
    INTERFACE_COMPILE_DEFINITIONS
      ARM_COMPUTE
    IMPORTED_LOCATION
      ${ARMComputeCore_LIBRARY}
    IMPORTED_NO_SONAME
      TRUE
  )
endif()


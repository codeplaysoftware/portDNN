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

# Try to find the portBLAS library.
#
# If the library is found then the `PORTBLAS::portblas` target will be
# exported with the required include directories.
#
# Sets the following variables:
#   portBLAS_FOUND        - whether the system has portBLAS
#   portBLAS_INCLUDE_DIRS - the portBLAS include directory

find_library(portBLAS_LIBRARY
  NAMES portBLAS libportblas
  PATH_SUFFIXES lib/lib
  HINTS ${portBLAS_DIR}
  DOC "The portBLAS shared library"
)

find_path(portBLAS_INCLUDE_DIR
  NAMES portblas.h
  PATH_SUFFIXES lib/include/portblas
  HINTS ${portBLAS_DIR}
  DOC "The portBLAS include directory"
)

find_path(portBlas_VPTR_INCLUDE_DIR
  NAMES vptr/virtual_ptr.hpp
  PATH_SUFFIXES external/computecpp-sdk/include
  HINTS ${portBLAS_DIR}
  DOC "The portBLAS virtual pointer include directory"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(portBLAS
  FOUND_VAR portBLAS_FOUND
  REQUIRED_VARS portBLAS_LIBRARY
                portBLAS_INCLUDE_DIR
                portBLAS_VPTR_INCLUDE_DIR
)

mark_as_advanced(portBLAS_FOUND
                 portBLAS_LIBRARY
                 portBLAS_VPTR_INCLUDE_DIR
                 portBLAS_INCLUDE_DIR
)

if(portBLAS_FOUND)
  set(portBLAS_INCLUDE_DIRS
    ${portBLAS_INCLUDE_DIR} ${portBLAS_VPTR_INCLUDE_DIR})
endif()

if(portBLAS_FOUND AND NOT TARGET PORTBLAS::portblas)
  add_library(PORTBLAS::portblas UNKNOWN IMPORTED)
  set_target_properties(PORTBLAS::portblas PROPERTIES
    IMPORTED_LOCATION "${portBLAS_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${portBLAS_INCLUDE_DIRS}"
  )
endif()

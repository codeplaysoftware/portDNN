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

# Try to find the sycl_blas library.
#
# If the library is found then the `SYCL_BLAS::sycl_blas` target will be
# exported with the required include directories.
#
# Sets the following variables:
#   sycl_blas_FOUND        - whether the system has sycl_blas
#   sycl_blas_INCLUDE_DIRS - the sycl_blas include directory

find_library(sycl_blas_LIBRARY
  NAMES sycl_blas libsycl_blas
  PATH_SUFFIXES lib/lib
  HINTS ${SyclBLAS_DIR}
  DOC "The SyclBLAS shared library"
)

find_path(SyclBLAS_INCLUDE_DIR
  NAMES sycl_blas.h
  PATH_SUFFIXES lib/include/sycl_blas
  HINTS ${SyclBLAS_DIR}
  DOC "The SyclBLAS include directory"
)

find_path(sycl_blas_VPTR_INCLUDE_DIR
  NAMES vptr/virtual_ptr.hpp
  PATH_SUFFIXES external/computecpp-sdk/include
  HINTS ${sycl_blas_DIR}
  DOC "The sycl_blas virtual pointer include directory"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(sycl_blas
  FOUND_VAR sycl_blas_FOUND
  REQUIRED_VARS sycl_blas_LIBRARY
                sycl_blas_INCLUDE_DIR
                sycl_blas_VPTR_INCLUDE_DIR
)

mark_as_advanced(sycl_blas_FOUND
                 sycl_blas_LIBRARY
                 sycl_blas_VPTR_INCLUDE_DIR
                 sycl_blas_INCLUDE_DIR
)

if(sycl_blas_FOUND)
  set(sycl_blas_INCLUDE_DIRS
    ${sycl_blas_INCLUDE_DIR} ${sycl_blas_VPTR_INCLUDE_DIR})
endif()

if(sycl_blas_FOUND AND NOT TARGET SYCL_BLAS::sycl_blas)
  add_library(SYCL_BLAS::sycl_blas UNKNOWN IMPORTED)
  set_target_properties(SYCL_BLAS::sycl_blas PROPERTIES
    IMPORTED_LOCATION "${sycl_blas_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${sycl_blas_INCLUDE_DIRS}"
  )
endif()

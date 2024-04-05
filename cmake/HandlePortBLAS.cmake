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
snn_include_guard(HANDLE_PORTBLAS)

if(NOT SNN_DOWNLOAD_PORTBLAS)
  find_package(portBLAS)
  if(NOT portBLAS_FOUND)
    find_package(portBLAS NO_MODULE)
  endif()
endif()

if(NOT portBLAS_FOUND AND (SNN_DOWNLOAD_PORTBLAS OR SNN_DOWNLOAD_MISSING_DEPS))
  include(ExternalProject)
  set(portBLAS_REPO "https://github.com/codeplaysoftware/sycl-blas.git" CACHE STRING
    "portBLAS git repository to clone"
  )
  set(portBLAS_GIT_TAG "8d29746fa2a33be6d126e177012f208337b3217c" CACHE STRING
    "Git tag, branch or commit to use for the portBLAS library"
  )

  set(portBLAS_SOURCE_DIR ${sycldnn_BINARY_DIR}/portBLAS-src)
  set(portBLAS_BINARY_DIR ${sycldnn_BINARY_DIR}/portBLAS-build)
  set(portBLAS_LIBNAME ${CMAKE_SHARED_LIBRARY_PREFIX}portblas${CMAKE_SHARED_LIBRARY_SUFFIX})
  set(portBLAS_LIBRARY ${portBLAS_BINARY_DIR}/${portBLAS_LIBNAME})
  set(portBLAS_BYPRODUCTS ${portBLAS_LIBRARY})
  if(CMAKE_CROSSCOMPILING)
    set(cmake_toolchain "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}")
  endif()
  if(NOT TARGET portBLAS_download)
    set(ComputeCpp_CMAKE_FLAGS "")
    set(DPCPP_CMAKE_FLAGS "")
    if (ComputeCpp_FOUND)
        set(ComputeCpp_CMAKE_FLAGS "-DComputeCpp_DIR=${ComputeCpp_DIR}")
    endif()
    if (DPCPP_FOUND)
        set(DPCPP_CMAKE_FLAGS "-DDPCPP_SYCL_TARGET=${SNN_DEVICE_TRIPLE}")
    endif()

    if(NOT "${SNN_DEVICE_TRIPLE}" STREQUAL "")
      set(PORTBLAS_TARGET_TRIPLE_OPT "-DDPCPP_SYCL_TARGET=${SNN_DEVICE_TRIPLE}")
      if(NOT "${SNN_DPCPP_ARCH}" STREQUAL "")
        set(PORTBLAS_ARCH_OPT "-DDPCPP_SYCL_ARCH=${SNN_DPCPP_ARCH}")
      endif()
    endif()

    ExternalProject_Add(portBLAS_download
      GIT_REPOSITORY    ${portBLAS_REPO}
      GIT_TAG           ${portBLAS_GIT_TAG}
      GIT_CONFIG        advice.detachedHead=false
      SOURCE_DIR        ${portBLAS_SOURCE_DIR}
      BINARY_DIR        ${portBLAS_BINARY_DIR}
      CMAKE_ARGS        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                        -DBUILD_SHARED_LIBS=ON
                        -DBLAS_ENABLE_TESTING=OFF
                        -DBLAS_ENABLE_BENCHMARK=OFF
                        -DBLAS_BUILD_SAMPLES=OFF
                        -DBLAS_ENABLE_CONST_INPUT=ON
                        ${PORTBLAS_TARGET_TRIPLE_OPT}
                        ${PORTBLAS_ARCH_OPT}
                        -DTUNING_TARGET=${SNN_TUNING_TARGET}
                        ${ComputeCpp_CMAKE_FLAGS}
                        ${DPCPP_CMAKE_FLAGS}
                        -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
                        ${cmake_toolchain}
      INSTALL_COMMAND   ""
      TEST_COMMAND      ""
      BUILD_BYPRODUCTS ${portBLAS_BYPRODUCTS}
    )
  endif()
  set(portBLAS_INCLUDE_DIR
    ${portBLAS_SOURCE_DIR}/include CACHE PATH
    "The portBLAS include directory" FORCE
  )
  set(portBLAS_VPTR_INCLUDE_DIR
    ${portBLAS_SOURCE_DIR}/external/computecpp-sdk/include CACHE PATH
    "The portBLAS virtual pointer include directory" FORCE
  )
  set(portBLAS_INCLUDE_DIRS ${portBLAS_INCLUDE_DIR} ${portBLAS_VPTR_INCLUDE_DIR})
  # Have to explicitly make the include directories to add it to the library
  # target. This will be filled with the headers at build time when the
  # library is downloaded.
  file(MAKE_DIRECTORY ${portBLAS_INCLUDE_DIR})
  file(MAKE_DIRECTORY ${portBLAS_VPTR_INCLUDE_DIR})

  if(NOT TARGET PORTBLAS::portblas)
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
    add_library(PORTBLAS::portblas SHARED IMPORTED)
    set_target_properties(PORTBLAS::portblas PROPERTIES
      IMPORTED_LOCATION "${portBLAS_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${portBLAS_INCLUDE_DIRS}"
    )
    add_dependencies(PORTBLAS::portblas portBLAS_download)
  endif()
  mark_as_advanced(portBLAS_REPO portBLAS_GIT_TAG)
endif()

if(NOT portBLAS_FOUND)
  message(FATAL_ERROR
    "Could not find portBLAS, consider setting SNN_DOWNLOAD_MISSING_DEPS")
endif()

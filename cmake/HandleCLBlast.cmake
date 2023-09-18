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
snn_include_guard(HANDLE_CLBLAST)

if(NOT SNN_DOWNLOAD_CLBLAST)
  find_package(CLBlast QUIET)
endif()

if(NOT CLBlast_FOUND AND (SNN_DOWNLOAD_CLBLAST OR SNN_DOWNLOAD_MISSING_DEPS))
  find_package(OpenCL REQUIRED)
  include(ExternalProject)
  set(CLBlast_REPO "https://github.com/CNugteren/CLBlast" CACHE STRING
    "CLBlast git repository to clone"
  )
  set(CLBlast_GIT_TAG "8433985" CACHE STRING
    "Git tag, branch or commit to use for the CLBlast library"
  )
  set(CLBlast_SOURCE_DIR ${portdnn_BINARY_DIR}/CLBlast-src)
  set(CLBlast_BINARY_DIR ${portdnn_BINARY_DIR}/CLBlast-build)
  set(CLBlast_LIBNAME ${CMAKE_STATIC_LIBRARY_PREFIX}clblast${CMAKE_STATIC_LIBRARY_SUFFIX})
  set(CLBlast_LIBRARIES ${CLBlast_BINARY_DIR}/${CLBlast_LIBNAME})
  set(CLBlast_BYPRODUCTS ${CLBlast_LIBRARIES})
  if(CMAKE_CROSSCOMPILING)
    set(cmake_toolchain
      "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
    )
  endif()
  if(NOT TARGET CLBlast_download)
    ExternalProject_Add(CLBlast_download
      GIT_REPOSITORY    ${CLBlast_REPO}
      GIT_TAG           ${CLBlast_GIT_TAG}
      GIT_CONFIG        advice.detachedHead=false
      SOURCE_DIR        ${CLBlast_SOURCE_DIR}
      BINARY_DIR        ${CLBlast_BINARY_DIR}
      CMAKE_ARGS        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                        -DBUILD_SHARED_LIBS=OFF
                        -DTUNERS=OFF
                        -DTESTS=OFF
                        -DSAMPLES=OFF
                        -DOPENCL=ON
                        ${cmake_toolchain}
      INSTALL_COMMAND   ""
      TEST_COMMAND      ""
      BUILD_BYPRODUCTS ${CLBlast_BYPRODUCTS}
    )
  endif()
  set(CLBlast_INCLUDE_DIR
    ${CLBlast_SOURCE_DIR}/include CACHE PATH
    "The CLBlast include directory" FORCE
  )
  set(CLBlast_INCLUDE_DIRS ${CLBlast_INCLUDE_DIR})
  # Have to explicitly make the include directories to add it to the library
  # target. This will be filled with the headers at build time when the
  # library is downloaded.
  file(MAKE_DIRECTORY ${CLBlast_INCLUDE_DIR})

  if(NOT TARGET clblast)
    add_library(clblast IMPORTED UNKNOWN)
    set_target_properties(clblast PROPERTIES
      IMPORTED_LOCATION ${CLBlast_LIBRARIES}
      INTERFACE_INCLUDE_DIRECTORIES ${CLBlast_INCLUDE_DIRS}
      INTERFACE_LINK_LIBRARIES OpenCL::OpenCL
    )
    add_dependencies(clblast CLBlast_download)
  endif()
  set(CLBlast_FOUND true)
  mark_as_advanced(CLBlast_REPO CLBlast_GIT_TAG CLBlast_INCLUDE_DIR)
endif()

if(NOT CLBlast_FOUND)
  message(FATAL_ERROR
    "Could not find CLBlast, consider setting SNN_DOWNLOAD_MISSING_DEPS")
endif()

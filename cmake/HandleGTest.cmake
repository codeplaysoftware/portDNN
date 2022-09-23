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


# Either finds or downloads googletest library
#
# Depending on the cmake options this will try to find the googletest library or
# download the library from github. If the library cannot be found or
# downloaded then the cmake configuration will fail, otherwise the library will
# be available in the `GTest::GTest` and `GTest::Main` target.
#
# Note: The googletest library requires the Threads library and so cmake may
# fail if it cannot find this.
cmake_minimum_required(VERSION 3.10.2)

include(SNNHelpers)
snn_include_guard(HANDLE_GTEST)

if(NOT SNN_DOWNLOAD_GTEST)
  find_package(GTest QUIET)
  if(GTEST_FOUND AND NOT TARGET GTest::GTest)
    # Older versions of cmake don't create a GTest::GTest target
    find_package(Threads REQUIRED)
    add_library(GTest::GTest IMPORTED UNKNOWN)
    set_target_properties(GTest::GTest PROPERTIES
      IMPORTED_LOCATION             ${GTEST_LIBRARIES}
      INTERFACE_LINK_LIBRARIES      "Threads::Threads"
      INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIRS}"
    )
  endif()
  if(GTEST_FOUND AND NOT TARGET GTest::Main)
    # Older versions of cmake don't create a GTest::Main target
    add_library(GTest::Main IMPORTED UNKNOWN)
    set_target_properties(GTest::Main PROPERTIES
      IMPORTED_LOCATION             ${GTEST_MAIN_LIBRARIES}
      INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIRS}"
    )
  endif()
endif()
if(SNN_DOWNLOAD_GTEST OR (NOT GTEST_FOUND AND SNN_DOWNLOAD_MISSING_DEPS))
  find_package(Threads REQUIRED)
  include(ExternalProject)
  set(GTEST_GIT_TAG "release-1.12.1" CACHE STRING
    "Git tag, branch or commit to use for googletest"
  )
  if(CMAKE_CROSSCOMPILING)
    set(cmake_toolchain
      "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
    )
  endif()
  set(GTEST_SOURCE_DIR ${snn_tests_BINARY_DIR}/googletest-src)
  set(GTEST_BINARY_DIR ${snn_tests_BINARY_DIR}/googletest-build)
  set(GTEST_LIBNAME ${CMAKE_STATIC_LIBRARY_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX})
  set(GTEST_MAIN_LIBNAME ${CMAKE_STATIC_LIBRARY_PREFIX}gtest_main${CMAKE_STATIC_LIBRARY_SUFFIX})
  set(GTEST_LIBRARIES ${GTEST_BINARY_DIR}/lib/${GTEST_LIBNAME})
  set(GTEST_MAIN_LIBRARIES ${GTEST_BINARY_DIR}/lib/${GTEST_MAIN_LIBNAME})
  set(GTEST_INCLUDE_DIR ${GTEST_SOURCE_DIR}/googletest/include)
  list(APPEND GTEST_BYPRODUCTS "${GTEST_LIBRARIES}")
  list(APPEND GTEST_BYPRODUCTS "${GTEST_MAIN_LIBRARIES}")
  ExternalProject_Add(googletest
    GIT_REPOSITORY    https://github.com/google/googletest.git
    GIT_TAG           ${GTEST_GIT_TAG}
    GIT_SHALLOW       ON
    GIT_CONFIG        advice.detachedHead=false
    SOURCE_DIR        ${GTEST_SOURCE_DIR}
    BINARY_DIR        ${GTEST_BINARY_DIR}
    CMAKE_ARGS        -Dgtest_force_shared_crt=ON
                      -DBUILD_SHARED_LIBS=OFF
                      -DBUILD_GMOCK=OFF
                      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                      -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                      -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
                      -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
                      ${cmake_toolchain}
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
    BUILD_BYPRODUCTS  ${GTEST_BYPRODUCTS}
  )

  # Have to explicitly make the include directory to add it to the library
  # target. This will be filled with the headers at build time when the
  # googletest library is downloaded.
  file(MAKE_DIRECTORY ${GTEST_INCLUDE_DIR})
  list(APPEND GTEST_INCLUDE_DIRS ${GTEST_INCLUDE_DIR})

  add_library(GTest::GTest IMPORTED STATIC)
  add_dependencies(GTest::GTest googletest)
  set_target_properties(GTest::GTest PROPERTIES
    IMPORTED_LOCATION             ${GTEST_LIBRARIES}
    INTERFACE_LINK_LIBRARIES      "Threads::Threads"
    INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIRS}"
  )
  add_library(GTest::Main IMPORTED STATIC)
  add_dependencies(GTest::Main googletest)
  set_target_properties(GTest::Main PROPERTIES
    IMPORTED_LOCATION             ${GTEST_MAIN_LIBRARIES}
    INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIRS}"
  )
  mark_as_advanced(GTEST_FOUND GTEST_INCLUDE_DIRS)
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(GTest
    DEFAULT_MSG
    GTEST_LIBRARIES GTEST_INCLUDE_DIRS GTEST_MAIN_LIBRARIES
  )
endif()
if(NOT GTEST_FOUND)
  message(FATAL_ERROR
    "Could NOT find GTest, consider setting SNN_DOWNLOAD_MISSING_DEPS")
endif()

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


# Either finds or downloads googletest library
#
# Depending on the cmake options this will try to find the googletest library or
# download the library from github. If the library cannot be found or
# downloaded then the cmake configuration will fail, otherwise the library will
# be available in the `GTest::GTest` and `GTest::Main` target.
#
# Note: The googletest library requires the Threads library and so cmake may
# fail if it cannot find this.
cmake_minimum_required(VERSION 3.2.2)

if(NOT SNN_DOWNLOAD_GTEST)
  find_package(GTest)
  if(GTEST_FOUND AND NOT TARGET GTest::GTest)
    # Older versions of cmake don't create a GTest::GTest target
    add_library(GTest::GTest IMPORTED UNKNOWN)
    set_target_properties(GTest::GTest PROPERTIES
      IMPORTED_LOCATION ${GTEST_LIBRARIES}
    )
    find_package(Threads REQUIRED)
    set_target_properties(GTest::GTest PROPERTIES
      INTERFACE_LINK_LIBRARIES "Threads::Threads"
    )
    set_target_properties(GTest::GTest PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIRS}"
    )
  endif()
  if(GTEST_FOUND AND NOT TARGET GTest::Main)
    # Older versions of cmake don't create a GTest::Main target
    add_library(GTest::Main IMPORTED STATIC)
    set_target_properties(GTest::Main PROPERTIES
      IMPORTED_LOCATION ${GTEST_MAIN_LIBRARIES}
    )
    set_target_properties(GTest::Main PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIRS}"
    )
  endif()
endif()
if(SNN_DOWNLOAD_GTEST OR (NOT GTEST_FOUND AND SNN_DOWNLOAD_MISSING_DEPS))
  message(STATUS "Downloading googletest library")
  configure_file(
    ${CMAKE_SOURCE_DIR}/cmake/GTestDownload.cmake.in
    googletest-download/CMakeLists.txt
  )
  execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${snn_tests_BINARY_DIR}/googletest-download
  )
  if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
  endif()
  execute_process(COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${snn_tests_BINARY_DIR}/googletest-download
  )
  if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
  endif()

  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  add_subdirectory(${snn_tests_BINARY_DIR}/googletest-src
                   ${snn_tests_BINARY_DIR}/googletest-build
                   EXCLUDE_FROM_ALL)

  mark_as_advanced(GTEST_FOUND)
  list(APPEND GTEST_LIBRARIES gtest)
  list(APPEND GTEST_MAIN_LIBRARIES gtest_main)
  list(APPEND GTEST_BOTH_LIBRARIES ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})
  list(APPEND GTEST_INCLUDE_DIRS ${gtest_SOURCE_DIR}/include)
  if(NOT TARGET GTest::GTest)
    add_library(GTest::GTest ALIAS ${GTEST_LIBRARIES})
  endif()
  if(NOT TARGET GTest::Main)
    add_library(GTest::Main ALIAS ${GTEST_MAIN_LIBRARIES})
  endif()
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

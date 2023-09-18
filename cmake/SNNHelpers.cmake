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

include(AddColourDiagnostics)

# Helpers macro to ensure a file is only included once, to avoid creating
# multiple identical targets or overriding variables
macro(snn_include_guard NAME)
  if(_SNN_${NAME}_INCLUDED)
    return()
  else()
    set(_SNN_${NAME}_INCLUDED ON)
  endif()
endmacro()

snn_include_guard(SNN_HELPERS)

# Helper functions to add targets to SYCL DNN
#
# Supported helper functions:
#   snn_test  - adds a test executable target and registers it with ctest
#   snn_bench - adds a benchmark executable and registers it with ctest
#   snn_object_library - adds an object library target
#
# Low level functions:
#   snn_executable - adds an executable target
#   snn_target     - adds common flags, features and libraries to a target
#
# These helper functions all take named parameters, detais of which can be
# found below. Typically they take a `TARGET` and a list of `SOURCES`, along
# with a list of `PUBLIC_LIBRARIES` to add to the target. As not all targets
# will need to be compiled with SYCL support, there is an option `WITH_SYCL` to
# specify that SYCL support is required. Any additional C++ compiler flags
# required can be added to the target with `CXX_OPTS`.
#
set(SNN_TEST_DEFAULT_TIMEOUT "30" CACHE STRING
  "Default timeout for tests (seconds)")
set(SNN_TEST_SHORT_TIMEOUT "30" CACHE STRING
  "Timeout for short tests (seconds)")
set(SNN_TEST_MODERATE_TIMEOUT "300" CACHE STRING
  "Timeout for moderate tests (seconds)")
set(SNN_TEST_LONG_TIMEOUT "900" CACHE STRING
  "Timeout for long tests (seconds)")
set(SNN_TEST_ETERNAL_TIMEOUT "3600" CACHE STRING
  "Timeout for eternal tests (seconds)")
mark_as_advanced(
  SNN_TEST_DEFAULT_TIMEOUT
  SNN_TEST_SHORT_TIMEOUT
  SNN_TEST_MODERATE_TIMEOUT
  SNN_TEST_LONG_TIMEOUT
  SNN_TEST_ETERNAL_TIMEOUT
)

# Check whether the option `${PREFIX}_${OPTION_NAME}` is set, and if so then
# set the variable `OUT_VAR` to the text `${OPTION_NAME}`. This is useful for
# forwarding options though mutliple layers of cmake_parse_arguments.
macro(snn_forward_option OUT_VAR PREFIX OPTION_NAME)
  # This constructs the variable name for the option, then sets
  # `_is_option_set` to the value of this option
  set(_is_option_set ${${PREFIX}_${OPTION_NAME}})
  if(_is_option_set)
    set(${OUT_VAR} "${OPTION_NAME}")
  else()
    set(${OUT_VAR} "")
  endif()
endmacro()

# Warn if there are unparsed arguments left over from cmake_parse_arguments.
# This likely indicates an error in the cmake files.
macro(snn_warn_unparsed_args PREFIX)
  if(${PREFIX}_UNPARSED_ARGUMENTS)
    message(WARNING "Unparsed arguments: ${${PREFIX}_UNPARSED_ARGUMENTS}")
  endif()
endmacro()

# Add a compile flag to the target if that flag is supported.
# Will check that the compiler supports `FLAG`, and if so add it to `TARGET`
# with the specified `MODE` (one of `PUBLIC`, `PRIVATE` or `INTERFACE`)
function(snn_add_compile_flag TARGET MODE FLAG)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(${FLAG} HAVE_${FLAG}_SUPPORT)
  if(HAVE_${FLAG}_SUPPORT)
    target_compile_options(${TARGET} ${MODE} ${FLAG})
  endif()
endfunction()

# snn_target helper function
# Adds the required links, include directories, SYCL support and flags to a
# given cmake target.
#
# WITH_SYCL: whether to compile the target for SYCL
# TARGET: name of the target
# KERNEL_SOURCES: source files containing SYCL kernels
# PUBLIC_LIBRARIES: library targets to add to the target's interface
# PRIVATE_LIBRARIES: library targets to use to compile the target
# PUBLIC_COMPILE_DEFINITIONS: compile definitions to add to the target
# PRIVATE_COMPILE_DEFINITIONS: compile definitions to add to the target
# CXX_OPTS: additional compile flags to add to the target
function(snn_target)
  set(options
    WITH_SYCL
    HIGH_MEM
    INSTALL
  )
  set(one_value_args
    TARGET
  )
  set(multi_value_args
    KERNEL_SOURCES
    PUBLIC_LIBRARIES
    PRIVATE_LIBRARIES
    PUBLIC_COMPILE_DEFINITIONS
    PRIVATE_COMPILE_DEFINITIONS
    CXX_OPTS
  )
  cmake_parse_arguments(SNN_TARGET
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  snn_warn_unparsed_args(SNN_TARGET)

  if(${SNN_TARGET_HIGH_MEM})
    set_property(TARGET ${SNN_TARGET_TARGET} PROPERTY
      JOB_POOL_COMPILE high_mem
    )
  endif()
  if((DEFINED SNN_TARGET_PUBLIC_LIBRARIES) OR
    (DEFINED SNN_TARGET_PRIVATE_LIBRARIES))
    # Set link libraries directly to work around limitations of object
    # libraries prior to cmake 3.12.
    set_property(TARGET ${SNN_TARGET_TARGET} APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES ${SNN_TARGET_PUBLIC_LIBRARIES}
    )
    set_property(TARGET ${SNN_TARGET_TARGET} APPEND PROPERTY
      LINK_LIBRARIES ${SNN_TARGET_PUBLIC_LIBRARIES} ${SNN_TARGET_PRIVATE_LIBRARIES}
    )
    if(("Eigen::Eigen" IN_LIST SNN_TARGET_PUBLIC_LIBRARIES) OR
      ("Eigen::Eigen" IN_LIST SNN_TARGET_PRIVATE_LIBRARIES))
      set_target_properties(${SNN_TARGET_TARGET}
        PROPERTIES COMPUTECPP_INCLUDE_AFTER 1)
    endif()
  endif()
  if((DEFINED SNN_TARGET_PUBLIC_COMPILE_DEFINITIONS) OR
    (DEFINED SNN_TARGET_PRIVATE_COMPILE_DEFINITIONS))
    target_compile_definitions(${SNN_TARGET_TARGET}
      PUBLIC  ${SNN_TARGET_PUBLIC_COMPILE_DEFINITIONS}
      PRIVATE ${SNN_TARGET_PRIVATE_COMPILE_DEFINITIONS}
    )
  endif()
  target_include_directories(${SNN_TARGET_TARGET}
    PUBLIC  $<INSTALL_INTERFACE:${include_dest}>
            $<BUILD_INTERFACE:${portdnn_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${portdnn_BINARY_DIR}>
    PRIVATE $<BUILD_INTERFACE:${portdnn_SOURCE_DIR}>
  )

  # Specify some C++11 features used widely across the library
  target_compile_features(${SNN_TARGET_TARGET} PUBLIC
    cxx_auto_type
    cxx_constexpr
    cxx_final
    cxx_lambdas
    cxx_static_assert
  )
  if(NOT MSVC)
    snn_add_compile_flag(${SNN_TARGET_TARGET} PRIVATE -Wall)
    snn_add_compile_flag(${SNN_TARGET_TARGET} PRIVATE -Wextra)
  endif()
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND
      CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
    # GCC 4.8 will warn when a struct is zero initialised but there are no
    # explicit initializers for all the struct members.
    target_compile_options(${SNN_TARGET_TARGET}
      PRIVATE -Wno-missing-field-initializers
    )
  endif()
  if(SNN_TARGET_CXX_OPTS)
    target_compile_options(${SNN_TARGET_TARGET} PUBLIC ${SNN_TARGET_CXX_OPTS})
  endif()
  snn_add_colour_diagnostics(${SNN_TARGET_TARGET})
  if (MSVC AND MSVC_VERSION GREATER 1909)
    # If /Zc:__cplusplus is not specified the compiler won't change the
    # value of the macro __cplusplus. Lots of existing code rely on the
    # value of this macro to determine whether or not some features are
    # available. Setting this option will make sure the macro gets the
    # correct value in Visual Studio versions 2017 and newer.
    # https://docs.microsoft.com/en-us/cpp/build/reference/zc-cplusplus
    target_compile_options(${SNN_TARGET_TARGET} PRIVATE /Zc:__cplusplus)
  endif()
  if(${SNN_TARGET_WITH_SYCL})
    set(SNN_TARGET_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY})
    add_sycl_to_target(
      TARGET     ${SNN_TARGET_TARGET}
      BINARY_DIR ${SNN_TARGET_BIN_DIR}/${SNN_TARGET_TARGET}.dir
      SOURCES    ${SNN_TARGET_KERNEL_SOURCES}
    )
  endif()
  if(${SNN_TARGET_INSTALL})
    install(
      TARGETS ${SNN_TARGET_TARGET}
      RUNTIME DESTINATION ${runtime_dest}
      LIBRARY DESTINATION ${library_dest}
      ARCHIVE DESTINATION ${library_dest}
      PUBLIC_HEADER DESTINATION ${include_dest}
    )
  endif()
endfunction()

# snn_executable helper function
# Adds an executable target with the specified sources, libraries and include
# directories. If SYCL support is requested then that is added to the target as
# well.
#
# WITH_SYCL: whether to compile the executable for SYCL
# INSTALL: whether to exclude the executable from installation rules
# TARGET: name of executable for the target
# SOURCES: source files for the executable
# KERNEL_SOURCES: source files containing SYCL kernels
# OBJECTS: object files to add to the executable
# PUBLIC_LIBRARIES: library targets to add to the target's interface
# PRIVATE_LIBRARIES: library targets to use to compile the target
# PUBLIC_COMPILE_DEFINITIONS: compile definitions to add to the target
# PRIVATE_COMPILE_DEFINITIONS: compile definitions to add to the target
# CXX_OPTS: additional compile flags to add to the target
function(snn_executable)
  set(options
    WITH_SYCL
    HIGH_MEM
    INSTALL
  )
  set(one_value_args
    TARGET
  )
  set(multi_value_args
    SOURCES
    KERNEL_SOURCES
    OBJECTS
    PUBLIC_LIBRARIES
    PRIVATE_LIBRARIES
    PUBLIC_COMPILE_DEFINITIONS
    PRIVATE_COMPILE_DEFINITIONS
    CXX_OPTS
  )
  cmake_parse_arguments(SNN_EXEC
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  snn_warn_unparsed_args(SNN_EXEC)
  list(APPEND SNN_EXEC_SOURCES ${SNN_EXEC_KERNEL_SOURCES})
  if(NOT "${SNN_EXEC_SOURCES}" STREQUAL "")
    # CMake cannot remove duplicates from an empty list
    list(REMOVE_DUPLICATES SNN_EXEC_SOURCES)
  endif()
  add_executable(${SNN_EXEC_TARGET}
    ${SNN_EXEC_SOURCES}
    ${SNN_EXEC_OBJECTS}
  )
  snn_forward_option(_WITH_SYCL SNN_EXEC WITH_SYCL)
  snn_forward_option(_HIGH_MEM SNN_EXEC HIGH_MEM)
  snn_forward_option(_INSTALL SNN_EXEC INSTALL)
  snn_target(
    ${_WITH_SYCL}
    ${_HIGH_MEM}
    ${_INSTALL}
    TARGET               ${SNN_EXEC_TARGET}
    KERNEL_SOURCES       ${SNN_EXEC_KERNEL_SOURCES}
    PUBLIC_LIBRARIES     ${SNN_EXEC_PUBLIC_LIBRARIES}
    PRIVATE_LIBRARIES    ${SNN_EXEC_PRIVATE_LIBRARIES}
    PUBLIC_COMPILE_DEFINITIONS ${SNN_EXEC_PUBLIC_COMPILE_DEFINITIONS}
    PRIVATE_COMPILE_DEFINITIONS ${SNN_EXEC_PRIVATE_COMPILE_DEFINITIONS}
    CXX_OPTS             ${SNN_EXEC_CXX_OPTS}
  )
endfunction()

# snn_test helper function
# Adds a test target with the specified sources, libraries and include
# directories. If SYCL support is requested then that is added to the target as
# well.
#
# Test timeouts are set depending on the SIZE of the test. The timeouts are
# loosely based on bazel's test timeouts.
#
# parameters
# WITH_SYCL: whether to compile the test for SYCL
# TARGET: target name prefix
# SIZE: size of the test (short/moderate/long/eternal)
# SOURCES: source files for the test
# KERNEL_SOURCES: source files containing SYCL kernels
# OBJECTS: object files to add to the test
# PUBLIC_LIBRARIES: targets and flags for linking phase
# CXX_OPTS: additional compile flags to add to the target
function(snn_test)
  set(options
    WITH_SYCL
    HIGH_MEM
  )
  set(one_value_args
    TARGET
    SIZE
  )
  set(multi_value_args
    SOURCES
    KERNEL_SOURCES
    OBJECTS
    PUBLIC_LIBRARIES
    CXX_OPTS
  )
  cmake_parse_arguments(SNN_TEST
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  snn_warn_unparsed_args(SNN_TEST)
  message(STATUS "Test target: ${SNN_TEST_TARGET}")
  set(_NAME ${SNN_TEST_TARGET}_test)
  snn_forward_option(_WITH_SYCL SNN_TEST WITH_SYCL)
  snn_forward_option(_HIGH_MEM SNN_TEST HIGH_MEM)
  if(SNN_INSTALL_TESTS)
    set(_INSTALL "INSTALL")
  endif()
  snn_executable(
    ${_WITH_SYCL}
    ${_HIGH_MEM}
    ${_INSTALL}
    TARGET               ${_NAME}_bin
    SOURCES              ${SNN_TEST_SOURCES}
    KERNEL_SOURCES       ${SNN_TEST_KERNEL_SOURCES}
    OBJECTS              ${SNN_TEST_OBJECTS}
    PUBLIC_LIBRARIES     ${SNN_TEST_PUBLIC_LIBRARIES}
    PRIVATE_LIBRARIES    GTest::GTest GTest::Main
    CXX_OPTS             ${SNN_TEST_CXX_OPTS}
  )
  add_test(NAME ${_NAME} COMMAND ${_NAME}_bin --gtest_output=xml:output/)
  set(_TIMEOUT ${SNN_TEST_DEFAULT_TIMEOUT})
  if(SNN_TEST_SIZE STREQUAL "short")
    set(_TIMEOUT ${SNN_TEST_SHORT_TIMEOUT})
  elseif(SNN_TEST_SIZE STREQUAL "moderate")
    set(_TIMEOUT ${SNN_TEST_MODERATE_TIMEOUT})
  elseif(SNN_TEST_SIZE STREQUAL "long")
    set(_TIMEOUT ${SNN_TEST_LONG_TIMEOUT})
  elseif(SNN_TEST_SIZE STREQUAL "eternal")
    set(_TIMEOUT ${SNN_TEST_ETERNAL_TIMEOUT})
  endif()
  set_tests_properties(${_NAME} PROPERTIES
    TIMEOUT ${_TIMEOUT}
  )
endfunction()

# snn_bench helper function
# Adds a benchmark target with the specified sources, libraries and include
# directories. If SYCL support is requested then that is added to the target as
# well.
#
# parameters
# WITH_SYCL: whether to compile the benchmark for SYCL
# TARGET: target name prefix
# SOURCES: source files for the benchmark
# KERNEL_SOURCES: source files containing SYCL kernels
# OBJECTS: object files to add to the benchmark
# PUBLIC_LIBRARIES: targets and flags for linking phase
# PUBLIC_COMPILE_DEFINITIONS: compile definitions to add to the target
# CXX_OPTS: additional compile flags to add to the target
function(snn_bench)
  set(options
    WITH_SYCL
  )
  set(one_value_args
    TARGET
  )
  set(multi_value_args
    SOURCES
    KERNEL_SOURCES
    OBJECTS
    PUBLIC_LIBRARIES
    PUBLIC_COMPILE_DEFINITIONS
    CXX_OPTS
  )
  cmake_parse_arguments(SNN_BENCH
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  snn_warn_unparsed_args(SNN_BENCH)
  message(STATUS "Bench target: ${SNN_BENCH_TARGET}")
  set(_NAME ${SNN_BENCH_TARGET}_bench)
  snn_forward_option(_WITH_SYCL SNN_BENCH WITH_SYCL)
  if(${SNN_INSTALL_BENCHMARKS})
    set(_INSTALL "INSTALL")
  endif()
  snn_executable(
    ${_WITH_SYCL}
    ${_INSTALL}
    TARGET               ${_NAME}_bin
    SOURCES              ${SNN_BENCH_SOURCES}
    KERNEL_SOURCES       ${SNN_BENCH_KERNEL_SOURCES}
    OBJECTS              ${SNN_BENCH_OBJECTS}
    PUBLIC_LIBRARIES     ${SNN_BENCH_PUBLIC_LIBRARIES}
    PRIVATE_LIBRARIES    benchmark::benchmark
    PUBLIC_COMPILE_DEFINITIONS ${SNN_BENCH_PUBLIC_COMPILE_DEFINITIONS}
    PRIVATE_COMPILE_DEFINITIONS
    CXX_OPTS             ${SNN_BENCH_CXX_OPTS}
  )
  add_test(
    NAME           ${_NAME}
    COMMAND        ${_NAME}_bin --benchmark_out=output/${_NAME}.csv
                                --benchmark_out_format=csv
    CONFIGURATIONS Benchmark
  )
  # Ensure that the benchmark output directory is made
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/output)
endfunction()

# snn_object_library helper function
# Adds an object library target with the specified sources, libraries and include
# directories. If SYCL support is requested then that is added to the target as
# well.
#
# Parameters:
# WITH_SYCL: whether to add SYCL to the target
# TARGET: object library target name
# SOURCES: source files for the object library
# KERNEL_SOURCES: source files containing SYCL kernels
# PUBLIC_LIBRARIES: targets and flags for linking phase
# PUBLIC_COMPILE_DEFINITIONS: definitions to add when compiling
# CXX_OPTS: additional compile flags to add to the target
function(snn_object_library)
  set(options
    WITH_SYCL
  )
  set(one_value_args
    TARGET
  )
  set(multi_value_args
    SOURCES
    KERNEL_SOURCES
    PUBLIC_LIBRARIES
    PUBLIC_COMPILE_DEFINITIONS
    CXX_OPTS
  )
  cmake_parse_arguments(SNN_OBJLIB
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  snn_warn_unparsed_args(SNN_OBJLIB)
  list(APPEND SNN_OBJLIB_SOURCES ${SNN_OBJLIB_KERNEL_SOURCES})
  list(REMOVE_DUPLICATES SNN_OBJLIB_SOURCES)
  add_library(${SNN_OBJLIB_TARGET} OBJECT ${SNN_OBJLIB_SOURCES})
  set_target_properties(${SNN_OBJLIB_TARGET} PROPERTIES
    POSITION_INDEPENDENT_CODE TRUE
    COMPILE_DEFINITIONS sycl_dnn_EXPORTS
  )
  snn_forward_option(_WITH_SYCL SNN_OBJLIB WITH_SYCL)
  snn_target(
    ${_WITH_SYCL}
    TARGET               ${SNN_OBJLIB_TARGET}
    KERNEL_SOURCES       ${SNN_OBJLIB_KERNEL_SOURCES}
    PUBLIC_LIBRARIES     ${SNN_OBJLIB_PUBLIC_LIBRARIES}
    PUBLIC_COMPILE_DEFINITIONS ${SNN_OBJLIB_PUBLIC_COMPILE_DEFINITIONS}
    CXX_OPTS             ${SNN_OBJLIB_CXX_OPTS}
  )
endfunction()

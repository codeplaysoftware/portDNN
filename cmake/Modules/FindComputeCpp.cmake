#.rst:
# FindComputeCpp
#---------------
#
#   Copyright Codeplay Software Ltd.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use these files except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

#########################
#  FindComputeCpp.cmake
#########################
#
#  Tools for finding and building with ComputeCpp.
#
#  User must define ComputeCpp_DIR pointing to the ComputeCpp installation.
#
#  Latest version of this file can be found at:
#    https://github.com/codeplaysoftware/computecpp-sdk

cmake_minimum_required(VERSION 3.10.2)
include(CheckCXXCompilerFlag)
include(FindPackageHandleStandardArgs)

set(COMPUTECPP_USER_FLAGS "" CACHE STRING "User flags for compute++")
separate_arguments(COMPUTECPP_USER_FLAGS)
mark_as_advanced(COMPUTECPP_USER_FLAGS)

set(COMPUTECPP_BITCODE "spirv64" CACHE STRING
  "Bitcode type to use as SYCL target in compute++")
mark_as_advanced(COMPUTECPP_BITCODE)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.20)
  # Policy enabling rewrites of paths in depfiles when using ninja
  cmake_policy(SET CMP0116 OLD)
endif()

set(SYCL_LANGUAGE_VERSION "2020" CACHE STRING "SYCL version to use. Defaults to 1.2.1.")

find_package(OpenCL QUIET)

# Find ComputeCpp package
set(computecpp_find_hint
  "${ComputeCpp_DIR}"
  "$ENV{COMPUTECPP_DIR}")

# Used for running executables on the host
set(computecpp_host_find_hint ${computecpp_find_hint})

if(CMAKE_CROSSCOMPILING)
  # ComputeCpp_HOST_DIR is used to find executables that are run on the host
  set(computecpp_host_find_hint
    "${ComputeCpp_HOST_DIR}"
    "$ENV{COMPUTECPP_HOST_DIR}"
    ${computecpp_find_hint}
  )
endif()

find_program(ComputeCpp_DEVICE_COMPILER_EXECUTABLE compute++
  HINTS ${computecpp_host_find_hint}
  PATH_SUFFIXES bin
  NO_SYSTEM_ENVIRONMENT_PATH)

find_program(ComputeCpp_INFO_EXECUTABLE computecpp_info
  HINTS ${computecpp_host_find_hint}
  PATH_SUFFIXES bin
  NO_SYSTEM_ENVIRONMENT_PATH)

find_library(COMPUTECPP_RUNTIME_LIBRARY
  NAMES ComputeCpp
  HINTS ${computecpp_find_hint}
  PATH_SUFFIXES lib
  DOC "ComputeCpp Runtime Library")

# Found the library, use only single hint from now on
get_filename_component(computecpp_library_path "${COMPUTECPP_RUNTIME_LIBRARY}" DIRECTORY)
get_filename_component(computecpp_find_hint "${computecpp_library_path}/.." ABSOLUTE)

find_library(COMPUTECPP_RUNTIME_LIBRARY_DEBUG
  NAMES ComputeCpp_d ComputeCpp
  HINTS ${computecpp_find_hint}
  PATH_SUFFIXES lib
  DOC "ComputeCpp Debug Runtime Library")

find_path(ComputeCpp_INCLUDE_DIRS
  NAMES "CL/sycl.hpp"
  HINTS ${computecpp_find_hint}/include
  DOC "The ComputeCpp include directory")
get_filename_component(ComputeCpp_INCLUDE_DIRS
  ${ComputeCpp_INCLUDE_DIRS} ABSOLUTE)

get_filename_component(computecpp_canonical_root_dir
  "${ComputeCpp_INCLUDE_DIRS}/.." ABSOLUTE)
set(ComputeCpp_ROOT_DIR "${computecpp_canonical_root_dir}" CACHE PATH
    "The root of the ComputeCpp install")

if(NOT ComputeCpp_INFO_EXECUTABLE)
  message(WARNING "Can't find computecpp_info - check ComputeCpp_DIR")
else()
  execute_process(COMMAND ${ComputeCpp_INFO_EXECUTABLE} "--dump-version"
    OUTPUT_VARIABLE ComputeCpp_VERSION
    RESULT_VARIABLE ComputeCpp_INFO_EXECUTABLE_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT ComputeCpp_INFO_EXECUTABLE_RESULT EQUAL "0")
    message(WARNING "computecpp - Error when obtaining version")
  endif()
  # The ComputeCpp_VERSION is set as something like "CE 1.0.3", so we first
  # need to extract the version number from the string.
  string(REGEX MATCH "([0-9]+\.[0-9]+\.[0-9]+)"
    ComputeCpp_VERSION ${ComputeCpp_VERSION})

  execute_process(COMMAND ${ComputeCpp_INFO_EXECUTABLE} "--dump-is-supported"
    OUTPUT_VARIABLE ComputeCpp_PLATFORM_IS_SUPPORTED
    RESULT_VARIABLE ComputeCpp_INFO_EXECUTABLE_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT ComputeCpp_INFO_EXECUTABLE_RESULT EQUAL "0")
    message(WARNING "computecpp - Error when checking platform support")
  elseif(NOT ComputeCpp_PLATFORM_IS_SUPPORTED)
    message(WARNING "computecpp - Your system might not support ComputeCpp")
  endif()
endif()

find_package_handle_standard_args(ComputeCpp
  REQUIRED_VARS ComputeCpp_ROOT_DIR
                ComputeCpp_DEVICE_COMPILER_EXECUTABLE
                ComputeCpp_INFO_EXECUTABLE
                COMPUTECPP_RUNTIME_LIBRARY
                COMPUTECPP_RUNTIME_LIBRARY_DEBUG
                ComputeCpp_INCLUDE_DIRS
                OpenCL_FOUND
  VERSION_VAR ComputeCpp_VERSION)
mark_as_advanced(ComputeCpp_ROOT_DIR
                 ComputeCpp_DEVICE_COMPILER_EXECUTABLE
                 ComputeCpp_INFO_EXECUTABLE
                 COMPUTECPP_RUNTIME_LIBRARY
                 COMPUTECPP_RUNTIME_LIBRARY_DEBUG
                 ComputeCpp_INCLUDE_DIRS
                 ComputeCpp_VERSION)

if(NOT ComputeCpp_FOUND)
  return()
endif()

set(_cached_CXX_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CXX_COMPILER ${ComputeCpp_DEVICE_COMPILER_EXECUTABLE})
check_cxx_compiler_flag("-sycl -sycl-std=${SYCL_LANGUAGE_VERSION}" _has_sycl_std)
# Cache entry is empty if check fails
if(NOT _has_sycl_std)
  set(_has_sycl_std 0)
endif()
set(CMAKE_CXX_COMPILER ${_cached_CXX_COMPILER})

set(COMPUTECPP_DEVICE_COMPILER_FLAGS -sycl)
list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS -O3 -DNDEBUG -mllvm -inline-threshold=1000)
mark_as_advanced(COMPUTECPP_DEVICE_COMPILER_FLAGS)

if(CMAKE_CROSSCOMPILING)
  if(NOT SNN_DONT_USE_TOOLCHAIN)
    list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS --gcc-toolchain=${SNN_TOOLCHAIN_DIR})
  endif()
  list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS --sysroot=${SNN_SYSROOT_DIR})
  list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS -target ${SNN_TARGET_TRIPLE})
endif()

foreach (bitcode IN ITEMS ${COMPUTECPP_BITCODE})
  if(NOT "${bitcode}" STREQUAL "")
    list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS -sycl-target ${bitcode})
  endif()
endforeach()

message(STATUS "compute++ flags - ${COMPUTECPP_DEVICE_COMPILER_FLAGS}")

if(CMAKE_COMPILER_IS_GNUCXX)
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8)
    message(FATAL_ERROR "host compiler - gcc version must be > 4.8")
  endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  if (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 3.6)
    message(FATAL_ERROR "host compiler - clang version must be > 3.6")
  endif()
endif()

if(MSVC)
  set(ComputeCpp_STL_CHECK_SRC __STL_check)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${ComputeCpp_STL_CHECK_SRC}.cpp
    "#include <CL/sycl.hpp>  \n"
    "int main() { return 0; }\n")
  set(_stl_test_command ${ComputeCpp_DEVICE_COMPILER_EXECUTABLE}
                        -sycl
                        ${COMPUTECPP_DEVICE_COMPILER_FLAGS}
                        -isystem ${ComputeCpp_INCLUDE_DIRS}
                        -isystem ${OpenCL_INCLUDE_DIRS}
                        -o ${ComputeCpp_STL_CHECK_SRC}.sycl
                        -c ${ComputeCpp_STL_CHECK_SRC}.cpp)
  execute_process(
    COMMAND ${_stl_test_command}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    RESULT_VARIABLE ComputeCpp_STL_CHECK_RESULT
    ERROR_VARIABLE ComputeCpp_STL_CHECK_ERROR_OUTPUT
    OUTPUT_QUIET)
  if(NOT ${ComputeCpp_STL_CHECK_RESULT} EQUAL 0)
    # Try disabling compiler version checks
    execute_process(
      COMMAND ${_stl_test_command}
              -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      RESULT_VARIABLE ComputeCpp_STL_CHECK_RESULT
      ERROR_VARIABLE ComputeCpp_STL_CHECK_ERROR_OUTPUT
      OUTPUT_QUIET)
    if(NOT ${ComputeCpp_STL_CHECK_RESULT} EQUAL 0)
      # Try again with __CUDACC__ and _HAS_CONDITIONAL_EXPLICIT=0. This relaxes the restritions in the MSVC headers
      execute_process(
        COMMAND ${_stl_test_command}
                -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
                -D_HAS_CONDITIONAL_EXPLICIT=0
                -D__CUDACC__
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        RESULT_VARIABLE ComputeCpp_STL_CHECK_RESULT
        ERROR_VARIABLE ComputeCpp_STL_CHECK_ERROR_OUTPUT
        OUTPUT_QUIET)
        if(NOT ${ComputeCpp_STL_CHECK_RESULT} EQUAL 0)
          message(FATAL_ERROR "compute++ cannot consume hosted STL headers. This means that compute++ can't \
                               compile a simple program in this platform and will fail when used in this system. \
                               \n ${ComputeCpp_STL_CHECK_ERROR_OUTPUT}")
        else()
          list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
                                                       -D_HAS_CONDITIONAL_EXPLICIT=0
                                                       -D__CUDACC__)
        endif()
    else()
      list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH)
    endif()
  endif()
  file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/${ComputeCpp_STL_CHECK_SRC}.cpp
              ${CMAKE_CURRENT_BINARY_DIR}/${ComputeCpp_STL_CHECK_SRC}.cpp.sycl)
endif(MSVC)

if(NOT TARGET ComputeCpp::ComputeCpp)
  add_library(ComputeCpp::ComputeCpp UNKNOWN IMPORTED)
  set_target_properties(ComputeCpp::ComputeCpp PROPERTIES
    IMPORTED_LOCATION_DEBUG          "${COMPUTECPP_RUNTIME_LIBRARY_DEBUG}"
    IMPORTED_LOCATION_RELWITHDEBINFO "${COMPUTECPP_RUNTIME_LIBRARY}"
    IMPORTED_LOCATION                "${COMPUTECPP_RUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES    "${ComputeCpp_INCLUDE_DIRS}"
    INTERFACE_COMPILE_DEFINITIONS    "CL_TARGET_OPENCL_VERSION=120"
    INTERFACE_LINK_LIBRARIES         "OpenCL::OpenCL"
  )
endif()

# This property allows targets to specify that their sources should be
# compiled with the integration header included after the user's
# sources, not before (e.g. when an enum is used in a kernel name, this
# is not technically valid SYCL code but can work with ComputeCpp)
define_property(
  TARGET PROPERTY COMPUTECPP_INCLUDE_AFTER
  BRIEF_DOCS "Include integration header after user source"
  FULL_DOCS "Changes compiler arguments such that the source file is
  actually the integration header, and the .cpp file is included on
  the command line so that it is seen by the compiler first. Enables
  non-standards-conformant SYCL code to compile with ComputeCpp."
)
define_property(
  TARGET PROPERTY INTERFACE_COMPUTECPP_FLAGS
  BRIEF_DOCS "Interface compile flags to provide compute++"
  FULL_DOCS  "Set additional compile flags to pass to compute++ when compiling
  any target which links to this one."
)
define_property(
  SOURCE PROPERTY COMPUTECPP_SOURCE_FLAGS
  BRIEF_DOCS "Source file compile flags for compute++"
  FULL_DOCS  "Set additional compile flags for compiling the SYCL integration
  header for the given source file."
)
define_property(
  TARGET PROPERTY SYCL_STANDARD
  BRIEF_DOCS "The version of SYCL to use for the target"
  FULL_DOCS "The version of SYCL to use for the target. Defaults to
  ${SYCL_LANGUAGE_VERSION} (can be set via the SYCL_LANGUAGE_VERSION
  option). Allowed values: 2017, 2020"
)

####################
#   __build_ir
####################
#
#  Adds a custom target for running compute++ and adding a dependency for the
#  resulting integration header and kernel binary.
#
#  TARGET : Name of the target.
#  SOURCE : Source file to be compiled.
#  BINARY_DIR : Intermediate directory to output the integration header.
#  COUNTER : Counter included in name of custom target. Different counter
#       values prevent duplicated names of custom target when source files with
#       the same name, but located in different directories, are used for the
#       same target.
#
function(__build_ir)
  set(options)
  set(one_value_args
    TARGET
    SOURCE
    BINARY_DIR
    COUNTER
  )
  set(multi_value_args)
  cmake_parse_arguments(ARG
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  get_filename_component(sourceFileName ${ARG_SOURCE} NAME)

  # Set the path to the integration header.
  # The .sycl filename must depend on the target so that different targets
  # using the same source file will be generated with a different rule.
  set(outputSyclFile ${ARG_BINARY_DIR}/${sourceFileName}.sycl)
  set(outputDeviceFile ${ARG_BINARY_DIR}/${sourceFileName}.bin)
  set(depfile_name ${outputSyclFile}.d)

  set(include_directories "$<TARGET_PROPERTY:${ARG_TARGET},INCLUDE_DIRECTORIES>")
  set(compile_definitions "$<TARGET_PROPERTY:${ARG_TARGET},COMPILE_DEFINITIONS>")
  set(generated_include_directories
    $<$<BOOL:${include_directories}>:-I$<JOIN:${include_directories},;-I>>)
  set(generated_compile_definitions
    $<$<BOOL:${compile_definitions}>:-D$<JOIN:${compile_definitions},;-D>>)

  # Obtain language standard of the file
  set(device_compiler_cxx_standard
    "-std=c++$<TARGET_PROPERTY:${ARG_TARGET},CXX_STANDARD>")
  get_target_property(sycl_std_version ${ARG_TARGET} SYCL_STANDARD)
  set(device_compiler_sycl_standard
    $<${_has_sycl_std}:"-sycl-std=${sycl_std_version}">)

  # Add any user-defined compiler options
  set(target_compile_flags "")
  get_target_property(target_compile_options
    ${ARG_TARGET} INTERFACE_COMPILE_OPTIONS)
  if(target_compile_options)
    list(APPEND target_compile_flags ${target_compile_options})
  endif()
  get_property(source_compile_flags
    SOURCE ${ARG_SOURCE}
    PROPERTY COMPUTECPP_SOURCE_FLAGS
  )
  if(source_compile_flags)
    list(APPEND target_compile_flags ${source_compile_flags})
  endif()

  # Copy compile options from libraries
  get_target_property(target_libraries ${ARG_TARGET} LINK_LIBRARIES)
  if(target_libraries)
    foreach(library ${target_libraries})
      if(TARGET ${library})
        get_target_property(lib_options ${library} INTERFACE_COMPILE_OPTIONS)
        if(lib_options)
          list(APPEND target_compile_flags ${lib_options})
        endif()
        get_target_property(ccpp_flags ${library} INTERFACE_COMPUTECPP_FLAGS)
        if(ccpp_flags)
          list(APPEND target_compile_flags ${ccpp_flags})
        endif()
      endif()
    endforeach()
  endif()

  list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS
    ${device_compiler_cxx_standard}
    ${device_compiler_sycl_standard}
    ${COMPUTECPP_USER_FLAGS}
    ${target_compile_flags})

  set(ir_dependencies ${ARG_SOURCE})
  if(target_libraries)
    foreach(library ${target_libraries})
      if(TARGET ${library})
        list(APPEND ir_dependencies ${library})
      endif()
    endforeach()
  endif()

  if(CMAKE_GENERATOR MATCHES "Ninja")
    file(RELATIVE_PATH rel_output_file ${CMAKE_BINARY_DIR} ${outputDeviceFile})
    set(generate_depfile -MMD -MF ${depfile_name} -MT ${rel_output_file})
    set(enable_depfile DEPFILE ${depfile_name})
  else()
    set(generate_depfile)
    set(enable_depfile)
  endif()

  # Add custom command for running compute++
  add_custom_command(
    OUTPUT ${outputDeviceFile} ${outputSyclFile}
    COMMAND ${ComputeCpp_DEVICE_COMPILER_EXECUTABLE}
            ${COMPUTECPP_DEVICE_COMPILER_FLAGS}
            "${generated_include_directories}"
            "${generated_compile_definitions}"
            -sycl-ih ${outputSyclFile}
            -o ${outputDeviceFile}
            -c ${ARG_SOURCE}
            ${generate_depfile}
    COMMAND_EXPAND_LISTS
    DEPENDS ${ir_dependencies}
    ${enable_depfile}
    WORKING_DIRECTORY ${ARG_BINARY_DIR}
    COMMENT "Building ComputeCpp integration header file ${outputSyclFile}")

  # Name: (user-defined name)_(source file)_(counter)_ih
  set(headerTargetName
    ${ARG_TARGET}_${sourceFileName}_${ARG_COUNTER}_ih)

  if(NOT MSVC)
    # Add a custom target for the generated integration header
    add_custom_target(${headerTargetName} DEPENDS ${outputSyclFile})
    add_dependencies(${ARG_TARGET} ${headerTargetName})
  endif()

  # This property can be set on a per-target basis to indicate that the
  # integration header should appear after the main source listing
  get_target_property(includeAfter ${ARG_TARGET} COMPUTECPP_INCLUDE_AFTER)

  if(includeAfter)
    # Change the source file to the integration header - e.g.
    # g++ -c source_file_name.cpp.sycl
    get_target_property(current_sources ${ARG_TARGET} SOURCES)
    # Remove absolute path to source file
    list(REMOVE_ITEM current_sources ${ARG_SOURCE})
    # Remove relative path to source file
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" ""
      rel_source_file ${ARG_SOURCE}
    )
    list(REMOVE_ITEM current_sources ${rel_source_file})
    # Add SYCL header to source list
    list(APPEND current_sources ${outputSyclFile})
    set_property(TARGET ${ARG_TARGET}
      PROPERTY SOURCES ${current_sources})
    # CMake/gcc don't know what language a .sycl file is, so tell them
    set_property(SOURCE ${outputSyclFile} PROPERTY LANGUAGE CXX)
    set(includedFile ${ARG_SOURCE})
    set(cppFile ${outputSyclFile})
  else()
    set_property(SOURCE ${outputSyclFile} PROPERTY HEADER_FILE_ONLY ON)
    set(includedFile ${outputSyclFile})
    set(cppFile ${ARG_SOURCE})
  endif()

  # Force inclusion of the integration header for the host compiler
  if(MSVC)
    # Group SYCL files inside Visual Studio
    source_group("SYCL" FILES ${outputSyclFile})

    if(includeAfter)
      # Allow the source file to be edited using Visual Studio.
      # It will be added as a header file so it won't be compiled.
      set_property(SOURCE ${ARG_SOURCE} PROPERTY HEADER_FILE_ONLY true)
    endif()

    # Add both source and the sycl files to the VS solution.
    target_sources(${ARG_TARGET} PUBLIC ${ARG_SOURCE} ${outputSyclFile})

    set(forceIncludeFlags "/FI${includedFile} /TP")
  else()
    set(forceIncludeFlags "-include ${includedFile} -x c++")
  endif()

  set_property(
    SOURCE ${cppFile}
    APPEND_STRING PROPERTY COMPILE_FLAGS "${forceIncludeFlags}"
  )

endfunction(__build_ir)

#######################
#  add_sycl_to_target
#######################
#
#  Adds a SYCL compilation custom command associated with an existing
#  target and sets a dependancy on that new command.
#
#  TARGET : Name of the target to add SYCL to.
#  BINARY_DIR : Intermediate directory to output the integration header.
#  SOURCES : Source files to be compiled for SYCL.
#
function(add_sycl_to_target)
  set(options)
  set(one_value_args
    TARGET
    BINARY_DIR
  )
  set(multi_value_args
    SOURCES
  )
  cmake_parse_arguments(ARG
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  set_target_properties(${ARG_TARGET} PROPERTIES LINKER_LANGUAGE CXX)
  get_target_property(TARGET_SYCL_VERSION ${ARG_TARGET} SYCL_STANDARD)
  if(NOT SYCL_STANDARD)
    set_target_properties(${ARG_TARGET}
      PROPERTIES SYCL_STANDARD ${SYCL_LANGUAGE_VERSION})
  endif()
  target_compile_definitions(${ARG_TARGET} PRIVATE
      "-DSYCL_LANGUAGE_VERSION=$<TARGET_PROPERTY:${ARG_TARGET},SYCL_STANDARD>")

  # If the CXX compiler is set to compute++ enable the driver.
  get_filename_component(cmakeCxxCompilerFileName "${CMAKE_CXX_COMPILER}" NAME)
  if("${cmakeCxxCompilerFileName}" STREQUAL "compute++")
    if(MSVC)
      message(FATAL_ERROR "The compiler driver is not supported by this system,
                           revert the CXX compiler to your default host compiler.")
    endif()

    get_target_property(includeAfter ${ARG_TARGET} COMPUTECPP_INCLUDE_AFTER)
    if(includeAfter)
      list(APPEND COMPUTECPP_USER_FLAGS -fsycl-ih-last)
    endif()
    list(INSERT COMPUTECPP_DEVICE_COMPILER_FLAGS 0 -sycl-driver)
    # Prepend COMPUTECPP_DEVICE_COMPILER_FLAGS and append COMPUTECPP_USER_FLAGS
    foreach(prop COMPILE_OPTIONS INTERFACE_COMPILE_OPTIONS)
      get_target_property(target_compile_options ${ARG_TARGET} ${prop})
      if(NOT target_compile_options)
        set(target_compile_options "")
      endif()
      set_property(
        TARGET ${ARG_TARGET}
        PROPERTY ${prop}
        ${COMPUTECPP_DEVICE_COMPILER_FLAGS}
        ${target_compile_options}
        ${COMPUTECPP_USER_FLAGS}
      )
    endforeach()
  else()
    set(file_counter 0)
    # Add custom target to run compute++ and generate the integration header
    foreach(source_file ${ARG_SOURCES})
      if(NOT IS_ABSOLUTE ${source_file})
        set(source_file "${CMAKE_CURRENT_SOURCE_DIR}/${source_file}")
      endif()
      __build_ir(
        TARGET     ${ARG_TARGET}
        SOURCE     ${source_file}
        BINARY_DIR ${ARG_BINARY_DIR}
        COUNTER    ${file_counter}
      )
      MATH(EXPR file_counter "${file_counter} + 1")
    endforeach()
  endif()

  # Link with the ComputeCpp runtime library.
  #
  # There are some types of target which do not allow the use of
  # target_link_libraries, such as object libraries and interface libraries. To
  # get around this we can explictly append ComputeCpp to the LINK_LIBRARIES
  # property.
  # This has the same effect as:
  # target_link_libraries(${ARG_TARGET}
  #   PUBLIC ComputeCpp::ComputeCpp
  # )
  set_property(TARGET ${ARG_TARGET} APPEND PROPERTY
    LINK_LIBRARIES ComputeCpp::ComputeCpp)
  set_property(TARGET ${ARG_TARGET} APPEND PROPERTY
    INTERFACE_LINK_LIBRARIES ComputeCpp::ComputeCpp)
endfunction(add_sycl_to_target)

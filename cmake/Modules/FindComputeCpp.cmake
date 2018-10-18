#.rst:
# FindComputeCpp
#---------------
#
#   Copyright 2018 Codeplay Software Ltd.
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

cmake_minimum_required(VERSION 3.2.2)
include(FindPackageHandleStandardArgs)

# Check that a supported host compiler can be found
if(CMAKE_COMPILER_IS_GNUCXX)
    # Require at least gcc 4.8
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8)
      message(FATAL_ERROR
        "host compiler - Not found! (gcc version must be at least 4.8)")
    endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # Require at least clang 3.6
    if (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 3.6)
      message(FATAL_ERROR
        "host compiler - Not found! (clang version must be at least 3.6)")
    endif()
endif()

set(COMPUTECPP_USER_FLAGS "" CACHE STRING "User flags for compute++")
mark_as_advanced(COMPUTECPP_USER_FLAGS)

option(COMPUTECPP_FGLRX_WORKAROUND
  "Linker workaround for fglrx AMD drivers to prevent deadlocks" OFF)
mark_as_advanced(COMPUTECPP_FGLRX_WORKAROUND)

set(COMPUTECPP_BITCODE "spir64" CACHE STRING
  "Bitcode type to use as SYCL target in compute++")
mark_as_advanced(COMPUTECPP_BITCODE)

# We only need the headers, so leave OpenCL optional and then check the include
# path variable is set.
find_package(OpenCL)
if(NOT OpenCL_INCLUDE_DIRS)
  message(FATAL_ERROR "Failed to locate OpenCL headers.")
endif()

# Find ComputeCpp package

if(DEFINED ComputeCpp_DIR)
  set(computecpp_find_hint ${ComputeCpp_DIR})
elseif(DEFINED ENV{COMPUTECPP_DIR})
  set(computecpp_find_hint $ENV{COMPUTECPP_DIR})
elseif(DEFINED COMPUTECPP_PACKAGE_ROOT_DIR)
  message(DEPRECATION
    "COMPUTECPP_PACKAGE_ROOT_DIR is not supported. Use ComputeCpp_DIR instead.")
  set(computecpp_find_hint ${COMPUTECPP_PACKAGE_ROOT_DIR})
endif()

# Used for running executables on the host
set(computecpp_host_find_hint ${computecpp_find_hint})

if(CMAKE_CROSSCOMPILING)
  # ComputeCpp_HOST_DIR is used to find executables that are run on the host
  if(DEFINED ComputeCpp_HOST_DIR)
    set(computecpp_host_find_hint ${ComputeCpp_HOST_DIR})
  elseif(DEFINED ENV{COMPUTECPP_HOST_DIR})
    set(computecpp_host_find_hint $ENV{COMPUTECPP_HOST_DIR})
  endif()
endif()

find_program(ComputeCpp_DEVICE_COMPILER_EXECUTABLE compute++
  PATHS ${computecpp_host_find_hint}
  PATH_SUFFIXES bin
  DOC "ComputeCpp device compiler")

find_program(ComputeCpp_INFO_EXECUTABLE computecpp_info
  PATHS ${computecpp_host_find_hint}
  PATH_SUFFIXES bin
  DOC "ComputeCpp Info tool")

find_library(COMPUTECPP_RUNTIME_LIBRARY
  NAMES ComputeCpp ComputeCpp_vs2015
  PATHS ${computecpp_find_hint}
  PATH_SUFFIXES lib
  DOC "ComputeCpp Runtime Library")

find_library(COMPUTECPP_RUNTIME_LIBRARY_DEBUG
  NAMES ComputeCpp ComputeCpp_vs2015_d
  PATHS ${computecpp_find_hint}
  PATH_SUFFIXES lib
  DOC "ComputeCpp Debug Runtime Library")

find_path(ComputeCpp_INCLUDE_DIRS
  NAMES "CL/sycl.hpp"
  PATHS ${computecpp_find_hint}/include
  DOC "The ComputeCpp include directory")
get_filename_component(ComputeCpp_INCLUDE_DIRS
  ${ComputeCpp_INCLUDE_DIRS} ABSOLUTE)

get_filename_component(computecpp_canonical_root_dir
  "${ComputeCpp_INCLUDE_DIRS}/.." ABSOLUTE)
set(ComputeCpp_ROOT_DIR "${computecpp_canonical_root_dir}" CACHE PATH
    "The root of the ComputeCpp install")

execute_process(COMMAND ${ComputeCpp_INFO_EXECUTABLE} "--dump-version"
  OUTPUT_VARIABLE ComputeCpp_VERSION
  RESULT_VARIABLE ComputeCpp_INFO_EXECUTABLE_RESULT
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT ComputeCpp_INFO_EXECUTABLE_RESULT EQUAL "0")
  message(FATAL_ERROR "Package version - Error obtaining version!")
endif()

execute_process(COMMAND ${ComputeCpp_INFO_EXECUTABLE} "--dump-is-supported"
  OUTPUT_VARIABLE ComputeCpp_PLATFORM_IS_SUPPORTED
  RESULT_VARIABLE ComputeCpp_INFO_EXECUTABLE_RESULT
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT ComputeCpp_INFO_EXECUTABLE_RESULT EQUAL "0")
  message(FATAL_ERROR "platform - Error checking platform support!")
else()
  if (ComputeCpp_PLATFORM_IS_SUPPORTED)
    message(STATUS "platform - your system can support ComputeCpp")
  else()
    message(WARNING "platform - your system CANNOT support ComputeCpp")
  endif()
endif()

find_package_handle_standard_args(ComputeCpp
  FOUND_VAR ComputeCpp_FOUND
  REQUIRED_VARS ComputeCpp_ROOT_DIR
                ComputeCpp_DEVICE_COMPILER_EXECUTABLE
                ComputeCpp_INFO_EXECUTABLE
                COMPUTECPP_RUNTIME_LIBRARY
                COMPUTECPP_RUNTIME_LIBRARY_DEBUG
                ComputeCpp_INCLUDE_DIRS
  VERSION_VAR ComputeCpp_VERSION)
mark_as_advanced(ComputeCpp_ROOT_DIR
                 ComputeCpp_DEVICE_COMPILER_EXECUTABLE
                 ComputeCpp_INFO_EXECUTABLE
                 COMPUTECPP_RUNTIME_LIBRARY
                 COMPUTECPP_RUNTIME_LIBRARY_DEBUG
                 ComputeCpp_INCLUDE_DIRS
                 ComputeCpp_VERSION
                 ComputeCpp_PLATFORM_IS_SUPPORTED)

if(NOT ComputeCpp_FOUND)
  return()
endif()

# Add optimisation and bitcode options to device compiler flags
list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS ${CMAKE_CXX_FLAGS_RELEASE})
list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS "-mllvm -inline-threshold=1000")
list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS "-sycl -sycl-target ${COMPUTECPP_BITCODE}")

if(CMAKE_CROSSCOMPILING)
  if(NOT SNN_DONT_USE_TOOLCHAIN)
    list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS --gcc-toolchain=${SNN_TOOLCHAIN_DIR})
  endif()
  list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS --sysroot=${SNN_SYSROOT_DIR})
  list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS -target ${SNN_TARGET_TRIPLE})
endif()

separate_arguments(COMPUTECPP_DEVICE_COMPILER_FLAGS)

if(NOT TARGET ComputeCpp::ComputeCpp)
  add_library(ComputeCpp::ComputeCpp UNKNOWN IMPORTED)
  set_target_properties(ComputeCpp::ComputeCpp PROPERTIES
    IMPORTED_LOCATION_DEBUG          "${COMPUTECPP_RUNTIME_LIBRARY_DEBUG}"
    IMPORTED_LOCATION_RELWITHDEBINFO "${COMPUTECPP_RUNTIME_LIBRARY_DEBUG}"
    IMPORTED_LOCATION                "${COMPUTECPP_RUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES    "${ComputeCpp_INCLUDE_DIRS};${OpenCL_INCLUDE_DIRS}"
    INTERFACE_COMPILE_DEFINITIONS    "CL_TARGET_OPENCL_VERSION=120"
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

####################
#   __build_ir
####################
#
#  Adds a custom target for running compute++ and adding a dependency for the
#  resulting integration header.
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
  cmake_parse_arguments(SNN_BUILD_IR
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  get_filename_component(sourceFileName ${SNN_BUILD_IR_SOURCE} NAME)

  # Set the path to the integration header.
  set(outputSyclFile ${SNN_BUILD_IR_BINARY_DIR}/${sourceFileName}.sycl)
  set(depfile_name ${outputSyclFile}.d)

  # Obtain language standard of the file
  set(device_compiler_cxx_standard)
  get_target_property(targetCxxStandard ${SNN_BUILD_IR_TARGET} CXX_STANDARD)
  if (targetCxxStandard MATCHES 17)
    set(device_compiler_cxx_standard "-std=c++1z")
  elseif (targetCxxStandard MATCHES 14)
    set(device_compiler_cxx_standard "-std=c++14")
  elseif (targetCxxStandard MATCHES 11)
    set(device_compiler_cxx_standard "-std=c++11")
  elseif (targetCxxStandard MATCHES 98)
    message(FATAL_ERROR "SYCL applications cannot be compiled using C++98")
  else ()
    set(device_compiler_cxx_standard "")
  endif()

  # Add any user-defined compiler options
  set(target_compile_flags "")
  get_target_property(target_compile_options
    ${SNN_BUILD_IR_TARGET} INTERFACE_COMPILE_OPTIONS
  )
  if(target_compile_options)
    list(APPEND target_compile_flags ${target_compile_options})
  endif()
  get_property(source_compile_flags
    SOURCE ${SNN_BUILD_IR_SOURCE}
    PROPERTY COMPUTECPP_SOURCE_FLAGS
  )
  if(source_compile_flags)
    list(APPEND target_compile_flags ${source_compile_flags})
  endif()

  # Copy compile options from libraries
  get_target_property(target_libraries ${SNN_BUILD_IR_TARGET} LINK_LIBRARIES)
  if(target_libraries)
    foreach(library ${target_libraries})
      get_target_property(lib_options ${library} INTERFACE_COMPILE_OPTIONS)
      if(lib_options)
        list(APPEND target_compile_flags ${lib_options})
      endif()
      get_target_property(ccpp_flags ${library} INTERFACE_COMPUTECPP_FLAGS)
      if(ccpp_flags)
        list(APPEND target_compile_flags ${ccpp_flags})
      endif()
    endforeach()
  endif()

  # Add any user-defined definitions to the device compiler command
  set(compile_definitions
    "$<TARGET_PROPERTY:${SNN_BUILD_IR_TARGET},COMPILE_DEFINITIONS>"
  )
  set(target_compile_definitions
    $<$<BOOL:${compile_definitions}>:-D$<JOIN:${compile_definitions},\t-D>>
  )

  # Add any user-defined include directories to the device compiler command
  set(include_directories
    "$<TARGET_PROPERTY:${SNN_BUILD_IR_TARGET},INCLUDE_DIRECTORIES>"
  )
  set(target_compile_includes
    $<$<BOOL:${include_directories}>:-I\"$<JOIN:${include_directories},\"\t-I\">\">
  )

  set(COMPUTECPP_DEVICE_COMPILER_FLAGS
    ${device_compiler_cxx_standard}
    ${COMPUTECPP_DEVICE_COMPILER_FLAGS}
    ${COMPUTECPP_USER_FLAGS}
    ${target_compile_flags}
    ${target_compile_definitions}
    ${target_compile_includes}
  )
  separate_arguments(COMPUTECPP_DEVICE_COMPILER_FLAGS)

  set(ir_dependencies ${SNN_BUILD_IR_SOURCE})
  if(target_libraries)
    foreach(library ${target_libraries})
      list(APPEND ir_dependencies ${library})
    endforeach()
  endif()

  # Depfile support was only added in CMake 3.7
  # CMake throws an error if it is unsupported by the generator (i. e. not ninja)
  if(CMAKE_GENERATOR MATCHES "Ninja" AND NOT CMAKE_VERSION VERSION_LESS 3.7.0)
    file(RELATIVE_PATH rel_output_file ${CMAKE_BINARY_DIR} ${outputSyclFile})
    set(generate_depfile -MMD -MF ${depfile_name} -MT ${rel_output_file})
    set(enable_depfile DEPFILE ${depfile_name})
  else()
    set(generate_depfile)
    set(enable_depfile)
  endif()

  # Add custom command for running compute++
  add_custom_command(
    OUTPUT ${outputSyclFile}
    COMMAND ${ComputeCpp_DEVICE_COMPILER_EXECUTABLE}
            ${COMPUTECPP_DEVICE_COMPILER_FLAGS}
            -o ${outputSyclFile}
            -c ${SNN_BUILD_IR_SOURCE}
            ${generate_depfile}
    DEPENDS ${ir_dependencies}
    IMPLICIT_DEPENDS CXX ${SNN_BUILD_IR_SOURCE}
    ${enable_depfile}
    WORKING_DIRECTORY ${SNN_BUILD_IR_BINARY_DIR}
    COMMENT "Building ComputeCpp integration header file ${outputSyclFile}")

  # Name: (user-defined name)_(source file)_(counter)_ih
  set(headerTargetName
    ${SNN_BUILD_IR_TARGET}_${sourceFileName}_${SNN_BUILD_IR_COUNTER}_ih)

  if(NOT MSVC)
    # Add a custom target for the generated integration header
    add_custom_target(${headerTargetName} DEPENDS ${outputSyclFile})
    add_dependencies(${SNN_BUILD_IR_TARGET} ${headerTargetName})
  endif()

  # This property can be set on a per-target basis to indicate that the
  # integration header should appear after the main source listing
  get_property(includeAfter TARGET ${SNN_BUILD_IR_TARGET}
      PROPERTY COMPUTECPP_INCLUDE_AFTER)

  if(includeAfter)
    # Change the source file to the integration header - e.g.
    # g++ -c source_file_name.cpp.sycl
    get_target_property(current_sources ${SNN_BUILD_IR_TARGET} SOURCES)
    # Remove absolute path to source file
    list(REMOVE_ITEM current_sources ${SNN_BUILD_IR_SOURCE})
    # Remove relative path to source file
    string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" ""
      rel_source_file ${SNN_BUILD_IR_SOURCE}
    )
    list(REMOVE_ITEM current_sources ${rel_source_file})
    # Add SYCL header to source list
    list(APPEND current_sources ${outputSyclFile})
    set_property(TARGET ${SNN_BUILD_IR_TARGET}
      PROPERTY SOURCES ${current_sources})
    # CMake/gcc don't know what language a .sycl file is, so tell them
    set_property(SOURCE ${outputSyclFile} PROPERTY LANGUAGE CXX)
    set(includedFile ${SNN_BUILD_IR_SOURCE})
    set(cppFile ${outputSyclFile})
  else()
    set(includedFile ${outputSyclFile})
    set(cppFile ${SNN_BUILD_IR_SOURCE})
  endif()

  # Force inclusion of the integration header for the host compiler
  if(MSVC)
    # Group SYCL files inside Visual Studio
    source_group("SYCL" FILES ${outputSyclFile})

    if(includeAfter)
      # Allow the source file to be edited using Visual Studio.
      # It will be added as a header file so it won't be compiled.
      set_property(SOURCE ${SNN_BUILD_IR_SOURCE} PROPERTY HEADER_FILE_ONLY true)
    endif()

    # Add both source and the sycl files to the VS solution.
    target_sources(${SNN_BUILD_IR_TARGET} PUBLIC ${SNN_BUILD_IR_SOURCE} ${outputSyclFile})

    set(forceIncludeFlags "/FI\"${includedFile}\" /TP")
  else()
      set(forceIncludeFlags "-include ${includedFile} -x c++ ")
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
  cmake_parse_arguments(SNN_ADD_SYCL
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  list(LENGTH SNN_ADD_SYCL_SOURCES _num_sources)
  if(${_num_sources} GREATER 0)
    set(file_counter 0)
    # Add custom target to run compute++ and generate the integration header
    foreach(source_file ${SNN_ADD_SYCL_SOURCES})
      if(NOT IS_ABSOLUTE ${source_file})
        set(source_file "${CMAKE_CURRENT_SOURCE_DIR}/${source_file}")
      endif()
      __build_ir(
        TARGET     ${SNN_ADD_SYCL_TARGET}
        SOURCE     ${source_file}
        BINARY_DIR ${SNN_ADD_SYCL_BINARY_DIR}
        COUNTER    ${file_counter}
      )
      MATH(EXPR file_counter "${file_counter} + 1")
    endforeach()
  endif()
  if(COMPUTECPP_FGLRX_WORKAROUND)
    # AMD's fglrx driver will potentially cause deadlocks in threaded programs
    # if the OpenCL library is linked after the standard library. In order to
    # workaround this we explicitly force the linker to link against OpenCL
    # before it links against ComputeCpp, rather than waiting until the OpenCL
    # library is required.
    set_property(TARGET ${SNN_ADD_SYCL_TARGET} APPEND PROPERTY
      LINK_LIBRARIES -Wl,--no-as-needed ${OpenCL_LIBRARIES} -Wl,--as-needed
    )
  endif()

  # Link with the ComputeCpp runtime library.
  #
  # There are some types of target which do not allow the use of
  # target_link_libraries, such as object libraries and interface libraries. To
  # get around this we can explictly append ComputeCpp to the LINK_LIBRARIES
  # property.
  # This has the same effect as:
  # target_link_libraries(${SNN_ADD_SYCL_TARGET}
  #   PUBLIC -Wl,--allow-shlib-undefined ComputeCpp::ComputeCpp
  # )
  set_property(TARGET ${SNN_ADD_SYCL_TARGET} APPEND PROPERTY
    LINK_LIBRARIES -Wl,--allow-shlib-undefined ComputeCpp::ComputeCpp
  )
  set_property(TARGET ${SNN_ADD_SYCL_TARGET} APPEND PROPERTY
    INTERFACE_LINK_LIBRARIES -Wl,--allow-shlib-undefined ComputeCpp::ComputeCpp
  )
endfunction(add_sycl_to_target)


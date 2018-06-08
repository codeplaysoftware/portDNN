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
#  User must define COMPUTECPP_PACKAGE_ROOT_DIR pointing to the ComputeCpp
#   installation.
#
#  Latest version of this file can be found at:
#    https://github.com/codeplaysoftware/computecpp-sdk
cmake_minimum_required(VERSION 3.2.2)

# Check that a supported host compiler can be found
if(CMAKE_COMPILER_IS_GNUCXX)
    # Require at least gcc 4.8
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8)
      message(FATAL_ERROR
        "host compiler - Not found! (gcc version must be at least 4.8)")
    else()
      message(STATUS "host compiler - gcc ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # Require at least clang 3.6
    if (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 3.6)
      message(FATAL_ERROR
        "host compiler - Not found! (clang version must be at least 3.6)")
    else()
      message(STATUS "host compiler - clang ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
endif()

option(COMPUTECPP_DISABLE_GCC_DUAL_ABI "Compile with pre-5.1 ABI" OFF)
mark_as_advanced(COMPUTECPP_DISABLE_GCC_DUAL_ABI)

set(COMPUTECPP_USER_FLAGS "" CACHE STRING "User flags for compute++")
mark_as_advanced(COMPUTECPP_USER_FLAGS)

option(COMPUTECPP_FGLRX_WORKAROUND
  "Linker workaround for fglrx AMD drivers to prevent deadlocks" OFF)
mark_as_advanced(COMPUTECPP_FGLRX_WORKAROUND)

set(COMPUTECPP_BITCODE "spir64" CACHE STRING
  "Bitcode type to use as SYCL target in compute++")
mark_as_advanced(COMPUTECPP_BITCODE)

# Platform-specific arguments
if(MSVC)
  # Workaround to an unfixed Clang bug, rationale:
  # https://github.com/codeplaysoftware/computecpp-sdk/pull/51#discussion_r139399093
  set (COMPUTECPP_PLATFORM_SPECIFIC_ARGS "-fno-ms-compatibility")
endif()

# Find OpenCL package
find_package(OpenCL REQUIRED)

# Find ComputeCpp package

# Try to read the environment variable
if(DEFINED ENV{COMPUTECPP_PACKAGE_ROOT_DIR})
  if(NOT COMPUTECPP_PACKAGE_ROOT_DIR)
    set(COMPUTECPP_PACKAGE_ROOT_DIR $ENV{COMPUTECPP_PACKAGE_ROOT_DIR})
  endif()
endif()

if(NOT COMPUTECPP_PACKAGE_ROOT_DIR)
  message(FATAL_ERROR
    "ComputeCpp package - Not found! (please set COMPUTECPP_PACKAGE_ROOT_DIR)")
else()
  message(STATUS "ComputeCpp package - Found")
endif()

# Obtain the path to compute++
find_program(COMPUTECPP_DEVICE_COMPILER compute++ PATHS
  ${COMPUTECPP_PACKAGE_ROOT_DIR} PATH_SUFFIXES bin NO_DEFAULT_PATH)
if (EXISTS ${COMPUTECPP_DEVICE_COMPILER})
  mark_as_advanced(COMPUTECPP_DEVICE_COMPILER)
  message(STATUS "compute++ - Found: ${COMPUTECPP_DEVICE_COMPILER}")
else()
  message(FATAL_ERROR "compute++ - Not found! (${COMPUTECPP_DEVICE_COMPILER})")
endif()

# Obtain the path to computecpp_info
find_program(COMPUTECPP_INFO_TOOL computecpp_info PATHS
  ${COMPUTECPP_PACKAGE_ROOT_DIR} PATH_SUFFIXES bin NO_DEFAULT_PATH)
if (EXISTS ${COMPUTECPP_INFO_TOOL})
  mark_as_advanced(${COMPUTECPP_INFO_TOOL})
  message(STATUS "computecpp_info - Found: ${COMPUTECPP_INFO_TOOL}")
else()
  message(FATAL_ERROR "computecpp_info - Not found! (${COMPUTECPP_INFO_TOOL})")
endif()

# Obtain the path to the ComputeCpp runtime library
find_library(COMPUTECPP_RUNTIME_LIBRARY
  NAMES ComputeCpp ComputeCpp_vs2015
  PATHS ${COMPUTECPP_PACKAGE_ROOT_DIR}
  HINTS ${COMPUTECPP_PACKAGE_ROOT_DIR}/lib PATH_SUFFIXES lib
  DOC "ComputeCpp Runtime Library" NO_DEFAULT_PATH)

if (EXISTS ${COMPUTECPP_RUNTIME_LIBRARY})
  mark_as_advanced(COMPUTECPP_RUNTIME_LIBRARY)
else()
  message(FATAL_ERROR "ComputeCpp Runtime Library - Not found!")
endif()

find_library(COMPUTECPP_RUNTIME_LIBRARY_DEBUG
  NAMES ComputeCpp ComputeCpp_vs2015_d
  PATHS ${COMPUTECPP_PACKAGE_ROOT_DIR}
  HINTS ${COMPUTECPP_PACKAGE_ROOT_DIR}/lib PATH_SUFFIXES lib
  DOC "ComputeCpp Debug Runtime Library" NO_DEFAULT_PATH)

if (EXISTS ${COMPUTECPP_RUNTIME_LIBRARY_DEBUG})
  mark_as_advanced(COMPUTECPP_RUNTIME_LIBRARY_DEBUG)
else()
  message(FATAL_ERROR "ComputeCpp Debug Runtime Library - Not found!")
endif()

# NOTE: Having two sets of libraries is Windows specific, not MSVC specific.
# Compiling with Clang on Windows would still require linking to both of them.
if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
  message(STATUS "ComputeCpp runtime (Release): ${COMPUTECPP_RUNTIME_LIBRARY} - Found")
  message(STATUS "ComputeCpp runtime  (Debug) : ${COMPUTECPP_RUNTIME_LIBRARY_DEBUG} - Found")
else()
  message(STATUS "ComputeCpp runtime: ${COMPUTECPP_RUNTIME_LIBRARY} - Found")
endif()

# Obtain the ComputeCpp include directory
set(COMPUTECPP_INCLUDE_DIRECTORY ${COMPUTECPP_PACKAGE_ROOT_DIR}/include)
get_filename_component(COMPUTECPP_INCLUDE_DIRECTORY ${COMPUTECPP_INCLUDE_DIRECTORY} ABSOLUTE)
if (NOT EXISTS ${COMPUTECPP_INCLUDE_DIRECTORY})
  message(FATAL_ERROR "ComputeCpp includes - Not found!")
else()
  message(STATUS "ComputeCpp includes - Found")
endif()

# Obtain the package version
execute_process(COMMAND ${COMPUTECPP_INFO_TOOL} "--dump-version"
  OUTPUT_VARIABLE COMPUTECPP_PACKAGE_VERSION
  RESULT_VARIABLE COMPUTECPP_INFO_TOOL_RESULT OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT COMPUTECPP_INFO_TOOL_RESULT EQUAL "0")
  message(FATAL_ERROR "Package version - Error obtaining version!")
else()
  mark_as_advanced(COMPUTECPP_PACKAGE_VERSION)
  message(STATUS "Package version - ${COMPUTECPP_PACKAGE_VERSION}")
endif()

# Check if the platform is supported
execute_process(COMMAND ${COMPUTECPP_INFO_TOOL} "--dump-is-supported"
  OUTPUT_VARIABLE COMPUTECPP_PLATFORM_IS_SUPPORTED
  RESULT_VARIABLE COMPUTECPP_INFO_TOOL_RESULT OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT COMPUTECPP_INFO_TOOL_RESULT EQUAL "0")
  message(FATAL_ERROR "platform - Error checking platform support!")
else()
  mark_as_advanced(COMPUTECPP_PLATFORM_IS_SUPPORTED)
  if (COMPUTECPP_PLATFORM_IS_SUPPORTED)
    message(STATUS "platform - your system can support ComputeCpp")
  else()
    message(STATUS "platform - your system CANNOT support ComputeCpp")
  endif()
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

if(NOT TARGET OpenCL::OpenCL)
  add_library(OpenCL::OpenCL IMPORTED UNKNOWN)
  set_target_properties(OpenCL::OpenCL PROPERTIES
    IMPORTED_LOCATION             "${OpenCL_LIBRARIES}"
    INTERFACE_INCLUDE_DIRECTORIES "${OpenCL_INCLUDE_DIRS}"
  )
endif()
add_library(ComputeCpp::ComputeCpp IMPORTED UNKNOWN)
set_target_properties(ComputeCpp::ComputeCpp PROPERTIES
  IMPORTED_LOCATION_DEBUG          "${COMPUTECPP_RUNTIME_LIBRARY_DEBUG}"
  IMPORTED_LOCATION_RELWITHDEBINFO "${COMPUTECPP_RUNTIME_LIBRARY_DEBUG}"
  IMPORTED_LOCATION                "${COMPUTECPP_RUNTIME_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES    "${COMPUTECPP_INCLUDE_DIRECTORY}"
  INTERFACE_LINK_LIBRARIES         "OpenCL::OpenCL"
)

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
  # Retrieve source file name.
  get_filename_component(sourceFileName ${SNN_BUILD_IR_SOURCE} NAME)

  # Set the path to the Sycl file.
  set(outputSyclFile ${SNN_BUILD_IR_BINARY_DIR}/${sourceFileName}.sycl)

  # Add any user-defined include to the device compiler
  target_include_directories(${SNN_BUILD_IR_TARGET} PRIVATE ${OpenCL_INCLUDE_DIRS})
  set(device_compiler_includes "")
  get_property(includeDirectories DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY
    INCLUDE_DIRECTORIES)
  foreach(directory ${includeDirectories})
    list(APPEND device_compiler_includes "-I${directory}")
  endforeach()
  get_target_property(targetIncludeDirectories ${SNN_BUILD_IR_TARGET} INCLUDE_DIRECTORIES)
  foreach(directory ${targetIncludeDirectories})
    list(APPEND device_compiler_includes "-I${directory}")
  endforeach()
  if (CMAKE_INCLUDE_PATH)
    foreach(directory ${CMAKE_INCLUDE_PATH})
      list(APPEND device_compiler_includes "-I${directory}")
    endforeach()
  endif()
  list(REMOVE_DUPLICATES device_compiler_includes)

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
  get_target_property(target_compile_definitions
    ${SNN_BUILD_IR_TARGET} INTERFACE_COMPILE_DEFINITIONS
  )
  if(target_compile_definitions)
    list(APPEND target_compile_flags ${target_compile_definitions})
  endif()
  get_property(source_compile_flags
    SOURCE ${SNN_BUILD_IR_SOURCE}
    PROPERTY COMPUTECPP_SOURCE_FLAGS
  )
  if(source_compile_flags)
    list(APPEND target_compile_flags ${source_compile_flags})
  endif()
  # Copy include directories, compile options and definitions from libraries
  get_target_property(target_libraries ${SNN_BUILD_IR_TARGET} LINK_LIBRARIES)
  if(target_libraries)
    foreach(library ${target_libraries})
      get_target_property(lib_includes ${library} INTERFACE_INCLUDE_DIRECTORIES)
      if(lib_includes)
        foreach(directory ${lib_includes})
          list(APPEND device_compiler_includes -isystem${directory})
        endforeach()
      endif()
      get_target_property(lib_options ${library} INTERFACE_COMPILE_OPTIONS)
      if(lib_options)
        list(APPEND target_compile_flags ${lib_options})
      endif()
      get_target_property(lib_defines ${library} INTERFACE_COMPILE_DEFINITIONS)
      if(lib_defines)
        foreach(define ${lib_defines})
          list(APPEND target_compile_flags -D${define})
        endforeach()
      endif()
      get_target_property(ccpp_flags ${library} INTERFACE_COMPUTECPP_FLAGS)
      if(ccpp_flags)
        list(APPEND target_compile_flags ${ccpp_flags})
      endif()
    endforeach()
  endif()

  set(COMPUTECPP_DEVICE_COMPILER_FLAGS
    ${device_compiler_cxx_standard}
    ${COMPUTECPP_DEVICE_COMPILER_FLAGS}
    ${COMPUTECPP_USER_FLAGS}
    ${target_compile_flags}
  )
  # Convert argument list format
  separate_arguments(COMPUTECPP_DEVICE_COMPILER_FLAGS)

  set(ir_dependencies ${SNN_BUILD_IR_SOURCE})
  if(target_libraries)
    foreach(library ${target_libraries})
      list(APPEND ir_dependencies ${library})
    endforeach()
  endif()

  # Add custom command for running compute++
  add_custom_command(
    OUTPUT ${outputSyclFile}
    COMMAND ${COMPUTECPP_DEVICE_COMPILER}
            ${COMPUTECPP_DEVICE_COMPILER_FLAGS}
            -isystem ${COMPUTECPP_INCLUDE_DIRECTORY}
            ${COMPUTECPP_PLATFORM_SPECIFIC_ARGS}
            ${device_compiler_includes}
            -o ${outputSyclFile}
            -c ${SNN_BUILD_IR_SOURCE}
    DEPENDS ${ir_dependencies}
    IMPLICIT_DEPENDS CXX ${SNN_BUILD_IR_SOURCE}
    WORKING_DIRECTORY ${SNN_BUILD_IR_BINARY_DIR}
    COMMENT "Building ComputeCpp integration header file ${outputSyclFile}")

  # Name:
  # (user-defined name)_(source file)_(counter)_ih
  set(headerTargetName
    ${SNN_BUILD_IR_TARGET}_${sourceFileName}_${SNN_BUILD_IR_COUNTER}_ih)

  if(NOT MSVC)
    # Add a custom target for the generated integration header
    add_custom_target(${headerTargetName} DEPENDS ${outputSyclFile})

    # Add a dependency on the integration header
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
      PROPERTY SOURCES ${current_sources}
    )
    # CMake/gcc don't know what language a .sycl file is, so tell them
    set_property(SOURCE ${outputSyclFile} PROPERTY LANGUAGE CXX)
    set(includedFile ${SNN_BUILD_IR_SOURCE})
    set(compiledFile ${outputSyclFile})
  else()
    set(includedFile ${outputSyclFile})
    set(compiledFile ${SNN_BUILD_IR_SOURCE})
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
    target_sources(${SNN_BUILD_IR_TARGET} PUBLIC ${SNN_BUILD_IR_TARGET} ${outputSyclFile})

    # NOTE: The Visual Studio generators parse compile flags differently,
    # hence the different argument syntax
    if(CMAKE_GENERATOR MATCHES "Visual Studio")
      set(forceIncludeFlags "/FI\"${includedFile}\" /TP")
    else()
      set(forceIncludeFlags /FI ${includedFile} /TP)
    endif()
  else()
      set(forceIncludeFlags "-include ${includedFile} -x c++")
  endif()
  # target_compile_options removes duplicated flags, which does not work
  # for -include (it should appear once per included file - CMake bug #15826)
  #   target_compile_options(${targetName} BEFORE PUBLIC ${forceIncludeFlags})
  # To avoid the problem, we get the value of the property and manually append
  # it to the previous status.
  get_property(current_flags
    SOURCE ${SNN_BUILD_IR_SOURCE}
    PROPERTY COMPILE_FLAGS
  )
  set_property(
    SOURCE ${compiledFile}
    PROPERTY COMPILE_FLAGS "${forceIncludeFlags} ${currentFlags}"
  )

  if(COMPUTECPP_DISABLE_GCC_DUAL_ABI)
    set_property(TARGET ${SNN_BUILD_IR_TARGET} APPEND PROPERTY COMPILE_DEFINITIONS
      "_GLIBCXX_USE_CXX11_ABI=0")
  endif()

endfunction(__build_ir)

#######################
#  add_sycl_to_target
#######################
#
#  Adds a SYCL compilation custom command associated with an existing
#  target and sets a dependancy on that new command.
#
#  TARGET : Name of the target to add a SYCL to.
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
    get_target_property(current_links ${SNN_ADD_SYCL_TARGET} LINK_LIBRARIES)
    list(APPEND current_links
      "-Wl,--no-as-needed -l${OpenCL_LIBRARIES} -Wl,--as-needed"
    )
    set_target_properties(${SNN_ADD_SYCL_TARGET} PROPERTIES
      LINK_LIBRARIES "${current_links}"
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
  get_target_property(_links ${SNN_ADD_SYCL_TARGET} LINK_LIBRARIES)
  list(APPEND _links
    -Wl,--allow-shlib-undefined ComputeCpp::ComputeCpp
  )
  get_target_property(_interface_links
    ${SNN_ADD_SYCL_TARGET} INTERFACE_LINK_LIBRARIES
  )
  list(APPEND _interface_links
    -Wl,--allow-shlib-undefined ComputeCpp::ComputeCpp
  )
  set_target_properties(${SNN_ADD_SYCL_TARGET} PROPERTIES
    LINK_LIBRARIES "${_links}"
    INTERFACE_LINK_LIBRARIES "${_interface_links}"
  )
endfunction(add_sycl_to_target)


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
#
set(DOXYGEN_SKIP_DOT FALSE)
find_package(Doxygen REQUIRED)
if(NOT ${DOXYGEN_DOT_FOUND})
  message(FATAL_ERROR
    "Building documentation requires dot/graphviz. Install graphviz, or set "
    "SNN_BUILD_DOCUMENTATION=OFF to disable the building of documentation.")
endif()

# Generate a valid Doxyfile for the project.
configure_file(Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)

# Build a list of dependencies. This is rather crude, and won't detect newly
# added files, but it's the best we can reasonably do with CMake
file(GLOB_RECURSE DOC_SOURCES "${PROJECT_SOURCE_DIR}/include/*.h")
list(APPEND DOC_SOURCES
  ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
  ${PROJECT_SOURCE_DIR}/README.md
  ${PROJECT_SOURCE_DIR}/CONTRIBUTING.md
  ${PROJECT_SOURCE_DIR}/docs/main.dox
)
# Build documentation
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/html/index.html
  COMMAND
    ${DOXYGEN_EXECUTABLE}
  ARGS
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
  WORKING_DIRECTORY
    ${portdnn_SOURCE_DIR}
  DEPENDS
    ${DOC_SOURCES}
  COMMENT
    "Generating API documentation with Doxygen"
  VERBATIM
)

add_custom_target(doc ALL
  DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/html/index.html
  COMMENT
    "Building API documentation with Doxygen"
  VERBATIM
)

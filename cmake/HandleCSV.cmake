# Copyright Codeplay Software Ltd
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

include(ExternalProject)
include(SNNHelpers)
snn_include_guard(HANDLE_CSV)

set(CSV_REPO "https://github.com/ben-strasser/fast-cpp-csv-parser" CACHE STRING
  "CSV Parser repository"
)
set(CSV_GIT_TAG "6636561" CACHE STRING
  "Commit-ish to check out in CSV repo"
)
set(CSV_SOURCE_DIR ${portdnn_BINARY_DIR}/fast-cpp-csv-parser)

if(NOT TARGET CSV_download)
  ExternalProject_Add(CSV_download
    GIT_REPOSITORY    ${CSV_REPO}
    GIT_TAG           ${CSV_GIT_TAG}
    GIT_CONFIG        advice.detachedHead=false
    SOURCE_DIR        ${CSV_SOURCE_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
  )
endif()
if(NOT TARGET CSV::fast-cpp-csv-parser)
  add_library(CSV::fast-cpp-csv-parser INTERFACE IMPORTED)
  set_target_properties(CSV::fast-cpp-csv-parser
    PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CSV_SOURCE_DIR}")
endif()
add_dependencies(CSV::fast-cpp-csv-parser CSV_download)

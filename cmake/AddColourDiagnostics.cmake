# Copyright Codeplay Software Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
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

option(SNN_FORCE_COLOUR_DIAGNOSTICS
  "Whether to force compilers to output errors with colours" ON)

# Helper function to add compiler flags forcing the compiler to report errors
# with terminal colour codes.
#
# This is especially useful when using the ninja build system, as that buffers
# compiler output and so compilers typically do not output colour diagnostics.
#
# target: The cmake target to add colour diagnostics to
#
function(snn_add_colour_diagnostics target)
  if(SNN_FORCE_COLOUR_DIAGNOSTICS)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
      target_compile_options(${target}
        PRIVATE -fdiagnostics-color=always
      )
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      target_compile_options(${target}
        PRIVATE -fcolor-diagnostics
      )
    endif()
  endif()
endfunction()

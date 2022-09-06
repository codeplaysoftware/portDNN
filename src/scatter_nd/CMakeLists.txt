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

cmake_minimum_required(VERSION 3.2.2)
include(SNNHelpers)

macro(generate_kernel out_var template)
  set(_filename "${INST_SCATTERND_FILENAME}_${DATA_TYPE}_${INDEX_TYPE}_${SCATTERND_TYPE}")
  set(_filename "${_filename}_${INDEX_DEPTH}_${VECTOR_WIDTH}.cc")
  set(_gen_file ${CMAKE_BINARY_DIR}/generated/scatter_nd/${_filename})
  configure_file(${template} ${_gen_file} @ONLY)
  list(APPEND ${out_var} ${_gen_file})
endmacro()

set(SCATTERND_INDICES_TYPES int32_t int64_t)
set(SCATTERND_DATA_TYPES float int32_t int64_t)
if(SNN_ENABLE_HALF)
  list(APPEND SCATTERND_DATA_TYPES cl::sycl::half)
endif()
if(SNN_ENABLE_DOUBLE)
  list(APPEND SCATTERND_DATA_TYPES double)
endif()

function(generate_scatter_nd)
  set(options)
  set(one_value_args
    OUTPUT_VAR
    FILENAME
  )
  set(multi_value_args)
  cmake_parse_arguments(INST_SCATTERND
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  set(_general_template queue_scatter_nd_kernel_impl.cc.in)
  set(_sources "")
  

  foreach(INDEX_TYPE IN LISTS SCATTERND_INDICES_TYPES)
    foreach(INDEX_DEPTH IN ITEMS 1 2 3 4)
      foreach(SCATTERND_TYPE IN ITEMS Assign Add Sub Mul Div)
        if(SCATTERND_TYPE STREQUAL "Assign" ) 
          foreach(DATA_TYPE IN LISTS SNN_DATA_INT_TYPES)
            foreach(VECTOR_WIDTH IN ITEMS 1 2 4)
              generate_kernel(_sources ${_general_template})
            endforeach()
          endforeach()
        else()
          set(VECTOR_WIDTH 1)
          foreach(DATA_TYPE IN LISTS SCATTERND_DATA_TYPES)
            generate_kernel(_sources ${_general_template})
          endforeach()
        endif()
      endforeach()
    endforeach()
  endforeach()
  set(${INST_SCATTERND_OUTPUT_VAR} ${_sources} PARENT_SCOPE)
endfunction()

generate_scatter_nd(
  OUTPUT_VAR scatter_nd_kernels
  FILENAME   scatter_nd
)

snn_object_library(
  WITH_SYCL
  TARGET 
    scatter_nd
  KERNEL_SOURCES
    ${scatter_nd_kernels}
  SOURCES
    launch_scatter_nd.cc
)

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

set(_backends)
set(_cxx_opts)
set(_softmax_providers)

if(SNN_TEST_SYCLBLAS)
  include(HandleSyclBLAS)
  list(APPEND _cxx_opts -DSNN_TEST_SYCLBLAS=1)
  list(APPEND _softmax_providers SyclBLAS::SyclBLAS)
  list(APPEND _backends SyclBLASBackend)
else()
  list(APPEND _backends SNNBackend)
endif()

macro(generate_kernel out_var template)
  set(_filename "${INST_SOFTMAX_FILENAME}_${DATA_TYPE}_${INDEX_TYPE}_${SOFTMAX_TYPE}")
  set(_filename "${_filename}_${BACKEND}_${VECTOR_WIDTH}.cc")
  set(_gen_file ${CMAKE_BINARY_DIR}/generated/softmax/${_filename})
  configure_file(${template} ${_gen_file} @ONLY)
  list(APPEND ${out_var} ${_gen_file})
endmacro()

function(generate_softmax)
  set(options)
  set(one_value_args
    OUTPUT_VAR
    FILENAME
  )
  set(multi_value_args)
  cmake_parse_arguments(INST_SOFTMAX
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  set(_general_template queue_softmax_kernel_impl.cc.in)
  set(_sources "")
  foreach(DATA_TYPE IN LISTS SNN_DATA_TYPES)
    foreach(INDEX_TYPE IN LISTS SNN_INDEX_TYPES)
      foreach(SOFTMAX_TYPE IN ITEMS Softmax)
        foreach(BACKEND IN LISTS _backends)
          foreach(VECTOR_WIDTH IN ITEMS 1 2 4)
            generate_kernel(_sources ${_general_template})
          endforeach()
        endforeach()
      endforeach()
    endforeach()
  endforeach()
  set(${INST_SOFTMAX_OUTPUT_VAR} ${_sources} PARENT_SCOPE)
endfunction()

generate_softmax(
  OUTPUT_VAR softmax_kernels
  FILENAME   softmax
)

snn_object_library(
  WITH_SYCL
  TARGET 
    softmax
  KERNEL_SOURCES
    ${softmax_kernels}
  SOURCES
    launch_softmax.cc
  PUBLIC_LIBRARIES
    ${_softmax_providers}
  CXX_OPTS
    ${_cxx_opts}
)

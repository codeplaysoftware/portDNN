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
include(SNNHelpers)

macro(generate_reduce_impl out_var)
  string(MAKE_C_IDENTIFIER ${DATA_TYPE} DTYPE_ID)
  string(TOLOWER ${OP} LOWER_OP)
  set(_filename "${GEN_REDUCE_FILENAME}_${LOWER_OP}_${DTYPE_ID}_${INDEX_TYPE}.cc")
  set(_gen_file ${CMAKE_BINARY_DIR}/generated/reduce/${_filename})
  configure_file(${GEN_REDUCE_TEMPLATE_FILE} ${_gen_file})
  list(APPEND ${out_var} ${_gen_file})
endmacro()

function(generate_reduce_kernels)
  set(options)
  set(one_value_args
    OUTPUT_VAR
    TEMPLATE_FILE
    FILENAME
  )
  set(multi_value_args)
  cmake_parse_arguments(GEN_REDUCE
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  set(_sources "")
  foreach(OP IN ITEMS Add Mean Max Min)
    foreach(DATA_TYPE IN LISTS SNN_DATA_TYPES)
      foreach(INDEX_TYPE IN LISTS SNN_INDEX_TYPES)
        generate_reduce_impl(_sources)
      endforeach()
    endforeach()
  endforeach()
  set(${GEN_REDUCE_OUTPUT_VAR} ${_sources} PARENT_SCOPE)
endfunction()

generate_reduce_kernels(
  OUTPUT_VAR    default_reduce_kernel_sources
  TEMPLATE_FILE queue_reduction_impl_default.cc.in
  FILENAME      default_reduce_kernel
)
set(SNN_ENABLE_SUBGROUPS 0)
if(ComputeCpp_VERSION VERSION_GREATER_EQUAL 2.11)
  generate_reduce_kernels(
    OUTPUT_VAR    subgroup_reduce_kernel_sources
    TEMPLATE_FILE queue_reduction_impl_subgroup.cc.in
    FILENAME      subgroup_reduce_kernel)
  set(SNN_ENABLE_SUBGROUPS 1)
endif()
snn_object_library(
  WITH_SYCL
  TARGET         reduce
  SOURCES        launch_reduction.cc
  KERNEL_SOURCES ${default_reduce_kernel_sources}
                 ${subgroup_reduce_kernel_sources}
)
target_compile_definitions(reduce
  PRIVATE -DSNN_ENABLE_SUBGROUPS=${SNN_ENABLE_SUBGROUPS})
set_target_properties(reduce PROPERTIES
  CXX_STANDARD 17
  SYCL_STANDARD 2020)

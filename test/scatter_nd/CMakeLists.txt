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
cmake_minimum_required(VERSION 3.2.2)

include(HandleGTest)
include(SNNHelpers)

set(_cxx_opts)
set(_scatter_providers)

snn_test(
  WITH_SYCL
  TARGET
    scatter_nd_test
  SOURCES
    scatter_nd_assign.cc
    scatter_nd_add.cc
    scatter_nd_advanced.cc
  PUBLIC_LIBRARIES
    sycl_dnn
  CXX_OPTS
    ${_cxx_opts}
)

if(SNN_ENABLE_USM)
  snn_test(
    WITH_SYCL
    TARGET
      scatter_nd_event_dependencies_test
    SOURCES
      scatter_nd_assign_event_dependencies.cc
      scatter_nd_add_event_dependencies.cc
      scatter_nd_advanced_event_dependencies.cc
    PUBLIC_LIBRARIES
      sycl_dnn
    CXX_OPTS
      ${_cxx_opts}
  )
endif()

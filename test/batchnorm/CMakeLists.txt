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
set(_batchnorm_providers)

if(SNN_TEST_SYCLBLAS)
  include(Handlesycl_blas)
  list(APPEND _cxx_opts -DSNN_TEST_SYCLBLAS=1)
  list(APPEND _batchnorm_providers SYCL_BLAS::sycl_blas)
endif()

snn_test(
  WITH_SYCL
  TARGET
    batchnorm_test
  SOURCES
    batchnorm_forward_Training.cc
    batchnorm_forward_Inference.cc
  OBJECTS
    $<TARGET_OBJECTS:batchnorm>
  PUBLIC_LIBRARIES
    ${_batchnorm_providers}
  CXX_OPTS
    ${_cxx_opts}
)
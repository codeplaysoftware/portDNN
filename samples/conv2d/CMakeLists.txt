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

if(SNN_INSTALL_SAMPLES)
  set(_install INSTALL)
else()
  set(_install)
endif()

snn_executable(
  WITH_SYCL
  TARGET
    conv2d
  KERNEL_SOURCES
    conv2d.cc
  PRIVATE_LIBRARIES
    sycl_dnn
    Eigen::Eigen
  ${_install}
)

# The onchip memory sample relies on the codeplay extension onchip_memory
# buffer property available in ComputeCpp 1.0.3 and onwards.
if (ComputeCpp_FOUND AND (ComputeCpp_VERSION VERSION_GREATER 1.0.2))
  snn_executable(
    WITH_SYCL
    TARGET
      on_chip_memory
    SOURCES
      on_chip_memory.cc
    PRIVATE_LIBRARIES
      sycl_dnn
      Eigen::Eigen
    ${_install}
  )
endif()

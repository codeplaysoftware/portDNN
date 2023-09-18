/*
 * Copyright Codeplay Software Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <memory>
#include <random>
#include <vector>

#include <CL/sycl.hpp>

#include <unsupported/Eigen/CXX11/Tensor>

#include "portdnn/backend/eigen_backend.h"
#include "portdnn/pooling/launch.h"
#include "portdnn/pooling/operators.h"
#include "portdnn/pooling/params.h"
#include "portdnn/status.h"

int main() {
  /* Default selectors behave in an implementation-defined manner, but will
   * return an OpenCL device. This is used to make a queue, device and then
   * backend, that will all be cleaned up automatically. */
  auto device_selector = cl::sycl::default_selector{};
  auto queue = std::unique_ptr<Eigen::QueueInterface>{
      new Eigen::QueueInterface{device_selector}};
  auto device = Eigen::SyclDevice{queue.get()};
  auto backend = sycldnn::backend::EigenBackend{device};

  /* This POD struct stores the parameters of the pooling operation. These
   * are very similar to the parameters used in the convolution operations. */
  sycldnn::pooling::PoolingParams params{};
  params.in_rows = 16;
  params.in_cols = 16;
  params.out_rows = 8;
  params.out_cols = 8;
  params.window_rows = 2;
  params.window_cols = 2;
  params.stride_rows = 2;
  params.stride_cols = 2;
  params.batch = 1;
  params.channels = 1;
  params.pad_rows = 0;
  params.pad_cols = 0;

  /* Device memory of the size of the tensor. */
  using value_type = float;
  auto input_nbytes = params.batch * params.in_rows * params.in_cols *
                      params.channels * sizeof(value_type);
  auto output_nbytes = params.batch * params.out_rows * params.out_cols *
                       params.channels * sizeof(value_type);

  auto* input_gpu_buffer =
      static_cast<value_type*>(device.allocate(input_nbytes));
  auto* output_gpu_buffer =
      static_cast<value_type*>(device.allocate(output_nbytes));

  /* Use random number generation to initialise the device memory. */
  std::mt19937 rng;
  std::uniform_real_distribution<value_type> dist(0, 20);
  std::vector<value_type> input(params.in_rows * params.in_cols);
  std::generate(input.begin(), input.end(), [&] { return dist(rng); });
  device.memcpyHostToDevice(input_gpu_buffer, input.data(), input_nbytes);

  /* Here the kernel is launched. The function returns an SNNStatus, which
   * is a combination of an event and error code. Here the event is waited
   * on to make sure that the kernel finishes before the data is copied back
   * from the device. */
  auto ev = sycldnn::pooling::launch<value_type, sycldnn::pooling::Max,
                                     sycldnn::pooling::Forward>(
      input_gpu_buffer, output_gpu_buffer, params, backend);

  ev.event.wait_and_throw();
  input.resize(params.out_rows * params.out_cols);
  device.memcpyDeviceToHost(
      input.data(), output_gpu_buffer,
      params.out_rows * params.out_cols * sizeof(value_type));

  return 0;
}

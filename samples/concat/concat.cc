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

#include "sycldnn/backend/snn_backend.h"

#include "sycldnn/concat/launch.h"
#include "sycldnn/concat/params.h"

#include "sycldnn/status.h"

#include <iostream>
#include <vector>

#include <CL/sycl.hpp>

#include <chrono>

using BufBackend = sycldnn::backend::SNNBackend;
using BufDeviceMem = BufBackend::pointer_type<float const>;
using BufProvider = sycldnn::backend::BackendProvider<BufBackend>;

BufDeviceMem copy_to_device_buf(std::vector<float>& input,
                                BufBackend& backend) {
  auto input_gpu = backend.allocate<float>(input.size());
  auto input_buf = input_gpu.get_buffer();
  auto event = backend.get_queue().submit([&](sycl::handler& cgh) {
    auto input_acc = input_buf.get_access<sycl::access::mode::write>(cgh);
    cgh.copy(input.data(), input_acc);
  });
  event.wait_and_throw();
  return input_gpu;
}

void copy_to_host_buf(std::vector<float>& output,
                      sycldnn::backend::DeviceMemPointer<float> output_gpu,
                      BufBackend& backend) {
  auto event = backend.get_queue().submit([&](sycl::handler& cgh) {
    auto acc =
        output_gpu.get_buffer().template get_access<sycl::access::mode::read>(
            cgh, sycl::range<1>{output.size()},
            sycl::id<1>{output_gpu.get_offset()});
    cgh.copy(acc, output.data());
  });
  event.wait_and_throw();
}

int main() {
  sycl::queue q([](sycl::exception_list l) {
    for (auto e : l) {
      try {
        std::rethrow_exception(e);
      } catch (sycl::exception& e) {
        std::cout << e.what() << " " << e.get_cl_code() << std::endl;
      }
    }
  });
  BufBackend backend(q);

  std::vector<float> x = {11, 21, 31, 41};
  BufDeviceMem x_gpu = copy_to_device_buf(x, backend);
  std::vector<float> y = {12, 13, 22, 23, 32, 33, 42, 43};
  BufDeviceMem y_gpu = copy_to_device_buf(y, backend);
  std::vector<float> z = {14, 24, 34, 44};
  BufDeviceMem z_gpu = copy_to_device_buf(z, backend);

  std::vector<BufDeviceMem> input = {x_gpu, y_gpu, z_gpu};

  std::vector<float> output_data(16, 0.0);
  auto output = backend.allocate<float>(16);

  sycldnn::concat::ConcatParams params;
  params.input_descriptors = {{1, 1, 4, 1}, {1, 1, 4, 2}, {1, 1, 4, 1}};
  params.rank = 4;
  params.axis = 3;

  auto concat_event = sycldnn::concat::launch<float, BufBackend>(
      input, output, params, backend);
  concat_event.event.wait_and_throw();

  auto output_dims = sycldnn::concat::calculate_concat_output_dims(params);

  copy_to_host_buf(output_data, output, backend);

  for (auto val : output_data) {
    std::cout << val << ", ";
  }
  std::cout << std::endl;
  for (auto dim : output_dims) {
    std::cout << dim << ", ";
  }
  std::cout << std::endl;

  return 0;
}

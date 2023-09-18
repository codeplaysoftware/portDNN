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

#include "portdnn/backend/snn_backend.h"

#include "portdnn/softmax/direction.h"
#include "portdnn/softmax/launch.h"
#include "portdnn/softmax/params.h"

#include <CL/sycl.hpp>
#include <iostream>

namespace snn = sycldnn;
#ifndef SYCL_IMPLEMENTATION_ONEAPI
namespace sycl = cl::sycl;
#endif

using Backend = snn::backend::SNNBackend;
using DeviceMem = Backend::pointer_type<float>;

int main() {
  sycl::queue q([](sycl::exception_list l) {
    for (auto e : l) {
      try {
        std::rethrow_exception(e);
      } catch (sycl::exception& e) {
        std::cout << e.what() << " " << e.get_cl_code() << "\n";
      }
    }
  });
  Backend backend(q);
  snn::softmax::SoftmaxParams params{};
  params.channels = 3;
  params.batch = 3;
  params.rows = 9;
  params.cols = 9;

  auto size = params.channels * params.rows * params.cols * params.batch;
  std::vector<float> in_vec(size, 1.0);
  std::vector<float> out_vec(size, 0.0);

  auto input_mem = backend.allocate<float>(size);
  auto output_mem = backend.allocate<float>(size);
  auto workspace = backend.allocate<float>(size / params.channels);
  auto buf_in = input_mem.get_buffer();
  auto event = q.submit([&](sycl::handler& cgh) {
    auto acc_in = buf_in.get_access<sycl::access::mode::write>(cgh);

    cgh.copy(in_vec.data(), acc_in);
  });
  event.wait_and_throw();

  auto st = std::chrono::high_resolution_clock::now();
  auto softmax_event = snn::softmax::launch<float, snn::softmax::Forward>(
      input_mem, workspace, output_mem, params, backend);
  softmax_event.event.wait_and_throw();

  softmax_event = snn::softmax::launch<float, snn::softmax::Gradient>(
      input_mem, input_mem, workspace, output_mem, params, backend);
  softmax_event.event.wait_and_throw();

  auto end = std::chrono::high_resolution_clock::now();

  std::cout << "Finished Execution of the Softmax event after time "
            << (end - st).count() << "ns\n\n";

  auto buf_out = output_mem.get_buffer();
  event = backend.get_queue().submit([&](sycl::handler& cgh) {
    auto acc_out = buf_out.get_access<sycl::access::mode::read>(cgh);

    cgh.copy(acc_out, out_vec.data());
  });
  event.wait_and_throw();

  backend.deallocate(input_mem);
  backend.deallocate(output_mem);
  backend.deallocate(workspace);
  return 0;
}

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

//#include "sycldnn/backend/sycl_blas_backend.h"
#include "sycldnn/backend/snn_backend.h"
//#include "sycldnn/backend/clblast_backend.h"

#include "sycldnn/conv2d/conv_type.h"
#include "sycldnn/conv2d/launch.h"
#include "sycldnn/conv2d/params.h"
#include "sycldnn/conv2d/selector/default_selector.h"
#include "sycldnn/conv2d/sizes.h"
#include "sycldnn/conv2d/workspace_size.h"

#include "sycldnn/helpers/padding.h"
#include "sycldnn/helpers/ratio.h"

#include "sycldnn/pointwise/launch.h"

#include "sycldnn/bias/launch.h"
#include "sycldnn/bias/params.h"

#include "sycldnn/transpose/launch.h"

#include "sycldnn/padding_mode.h"
#include "sycldnn/status.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include <CL/sycl.hpp>
#include <SYCL/codeplay.hpp>

namespace snn = sycldnn;
namespace sycl = cl::sycl;

// using Backend = snn::backend::SyclBLASBackend;
using Backend = snn::backend::SNNBackend;
// using Backend = snn::backend::CLBlastBackend;
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
  snn::bias::BiasParams bias_params{};
  bias_params.channels = 16;
  bias_params.batch = 1;
  bias_params.in_rows = 16;
  bias_params.in_cols = 16;
  bias_params.bias = 16;

  std::vector<float> in_(4096, 10.0);
  std::vector<float> bias_(16, 0.5);
  std::vector<float> out_(4096, 0.0);

  auto input_ = backend.allocate<float>(in_.size() * sizeof(float));
  auto biases_ = backend.allocate<float>(bias_.size() * sizeof(float));
  auto output_ = backend.allocate<float>(out_.size() * sizeof(float));
  auto buf_in = input_.get_buffer();
  auto buf_bias = biases_.get_buffer();
  auto buf_out = output_.get_buffer();
  auto event = backend.get_queue().submit([&](sycl::handler& cgh) {
    auto acc_in = buf_in.get_access<sycl::access::mode::write>(cgh);
    auto acc_bias = buf_bias.get_access<sycl::access::mode::write>(cgh);

    cgh.copy(in_.data(), acc_in);
    cgh.copy(bias_.data(), acc_bias);
  });
  event.wait_and_throw();

  auto st = std::chrono::high_resolution_clock::now();
  auto bias_event =
      snn::bias::launch<float>(input_, biases_, output_, bias_params, backend);
  bias_event.event.wait_and_throw();
  auto end = std::chrono::high_resolution_clock::now();

  std::cout << "Finished Execution of the Bias-Add event after time "
            << (end - st).count() << "ns\n\n";
  return 0;
}

/*
 * Copyright Codeplay Software Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use these files except in compliance with the License.
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
#ifndef PORTDNN_TEST_TRANSPOSE_TRANSPOSE_FIXTURE_H_
#define PORTDNN_TEST_TRANSPOSE_TRANSPOSE_FIXTURE_H_

#include <gtest/gtest.h>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <CL/sycl.hpp>

#include "portdnn/backend/snn_backend.h"
#include "portdnn/backend/snn_usm_backend.h"

#include "portdnn/helpers/scope_exit.h"

#include "portdnn/transpose/launch.h"

#include "test/backend/backend_test_fixture.h"
#include "test/gen/iota_initialised_data.h"
#include "test/helpers/dependency_check.h"

enum transpose_func { launch, nhwc_to_nchw, nchw_to_nhwc };

template <typename T>
struct TransposeEventFixture
    : public BackendTestFixture<sycldnn::backend::SNNUSMBackend> {
  using DataType = T;

 protected:
  void run(std::vector<int> const& sizes, std::vector<int> const& permutation,
           DataType max_val, transpose_func func) {
    const size_t tensor_size =
        std::accumulate(begin(sizes), end(sizes), 1, std::multiplies<int>());

    std::vector<DataType> in_data = iota_initialised_data(tensor_size, max_val);
    std::vector<DataType> out_data =
        iota_initialised_data(tensor_size, max_val);

    auto& provider = this->provider_;
    auto& backend = provider.get_backend();

    {
      auto in_gpu =
          provider.get_initialised_device_memory(tensor_size, in_data);
      auto out_gpu =
          provider.get_initialised_device_memory(tensor_size, out_data);

      SNN_ON_SCOPE_EXIT {
        provider.deallocate_ptr(in_gpu);
        provider.deallocate_ptr(out_gpu);
      };

      try {
        dependency_test_params dep_test_params;
        cl::sycl::event dependee_e = create_event(backend, dep_test_params);

        sycldnn::SNNStatus status;
        if (func == launch) {
          status = sycldnn::transpose::launch<DataType>(
              in_gpu, out_gpu, sizes, permutation, backend,
              std::vector<cl::sycl::event>{dependee_e});
        } else if (func == nhwc_to_nchw) {
          status = sycldnn::transpose::convert_nhwc_to_nchw<DataType>(
              in_gpu, out_gpu, sizes, backend,
              std::vector<cl::sycl::event>{dependee_e});
        } else if (func == nchw_to_nhwc) {
          status = sycldnn::transpose::convert_nchw_to_nhwc<DataType>(
              in_gpu, out_gpu, sizes, backend,
              std::vector<cl::sycl::event>{dependee_e});
        } else {
          FAIL() << "Unexpected func value";
        }

        check_dependency(dependee_e, status.event, backend, dep_test_params);

      } catch (cl::sycl::exception const& e) {
        throw std::runtime_error(e.what());
      }
    }
  }
};
#endif  // PORTDNN_TEST_TRANSPOSE_TRANSPOSE_FIXTURE_H_

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

#ifndef PORTDNN_TEST_POOLING_POOLING_FIXTURE_H_
#define PORTDNN_TEST_POOLING_POOLING_FIXTURE_H_

#include <gtest/gtest.h>

#include "portdnn/padding_mode.h"

#include "portdnn/helpers/scope_exit.h"

#include "portdnn/roi_align/launch.h"
#include "portdnn/roi_align/operators.h"
#include "portdnn/roi_align/params.h"

#include "test/backend/backend_test_fixture.h"
#include "test/gen/iota_initialised_data.h"
#include "test/helpers/float_comparison.h"

#include <numeric>
#include <string>
#include <vector>

template <typename DType, typename BatchIndicesType, typename Backend>
struct RoiAlignFixture : public BackendTestFixture<Backend> {
  using DataType = DType;
#ifdef SNN_USE_INT64
  using IndexType = int64_t;
#else
  using IndexType = int32_t;
#endif

  static unsigned constexpr max_ulps = 4u;

  void verify_output(std::vector<DataType> const& expected,
                     std::vector<DataType> const& actual) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      SCOPED_TRACE("Element: " + std::to_string(i));
      SNN_ALMOST_EQUAL(expected[i], actual[i], max_ulps);
    }
  }

  void test_roi_align(std::vector<DataType> const& rois,
                      std::vector<BatchIndicesType> const& batch_indices,
                      std::vector<DataType> const& expected_data_max_pool,
                      std::vector<DataType> const& expected_data_avg_pool,
                      sycldnn::roi_align::RoiAlignParams const& params) {
    const std::vector<IndexType> X_shape = {params.batch, params.channels,
                                            params.in_width, params.in_height};
    const IndexType X_num_elems = std::accumulate(
        X_shape.begin(), X_shape.end(), 1, std::multiplies<IndexType>());
    std::vector<DataType> X(X_num_elems);
    std::iota(X.begin(), X.end(), DataType(0));

    std::vector<DataType> out_data_max_pool(expected_data_max_pool.size(),
                                            DataType(0));
    std::vector<DataType> out_data_avg_pool(expected_data_avg_pool.size(),
                                            DataType(0));

    auto& provider = this->provider_;
    auto& backend = provider.get_backend();

    {
      auto X_gpu = provider.get_initialised_device_memory(X.size(), X);
      auto rois_gpu = provider.get_initialised_device_memory(rois.size(), rois);
      auto batch_indices_gpu = provider.get_initialised_device_memory(
          batch_indices.size(), batch_indices);
      auto out_gpu_max_pool = provider.get_initialised_device_memory(
          expected_data_max_pool.size(), out_data_max_pool);
      auto out_gpu_avg_pool = provider.get_initialised_device_memory(
          expected_data_avg_pool.size(), out_data_avg_pool);
      SNN_ON_SCOPE_EXIT {
        provider.deallocate_ptr(X_gpu);
        provider.deallocate_ptr(rois_gpu);
        provider.deallocate_ptr(batch_indices_gpu);
        provider.deallocate_ptr(out_gpu_max_pool);
        provider.deallocate_ptr(out_gpu_avg_pool);
      };

      try {
        // Max pooling
        auto status = sycldnn::roi_align::launch<DataType, BatchIndicesType,
                                                 sycldnn::roi_align::MaxPool>(
            X_gpu, rois_gpu, batch_indices_gpu, out_gpu_max_pool, params,
            backend);
        ASSERT_EQ(sycldnn::StatusCode::OK, status.status);
        status.event.wait_and_throw();

        // Average pooling
        status = sycldnn::roi_align::launch<DataType, BatchIndicesType,
                                            sycldnn::roi_align::AveragePool>(
            X_gpu, rois_gpu, batch_indices_gpu, out_gpu_avg_pool, params,
            backend);
        ASSERT_EQ(sycldnn::StatusCode::OK, status.status);
        status.event.wait_and_throw();
      } catch (cl::sycl::exception const& e) {
        throw std::runtime_error(e.what());
      }

      provider.copy_device_data_to_host(expected_data_max_pool.size(),
                                        out_gpu_max_pool, out_data_max_pool);
      provider.copy_device_data_to_host(expected_data_avg_pool.size(),
                                        out_gpu_avg_pool, out_data_avg_pool);
    }

    verify_output(expected_data_max_pool, out_data_max_pool);
    verify_output(expected_data_avg_pool, out_data_avg_pool);
  }
};

#endif  // PORTDNN_TEST_POOLING_POOLING_FIXTURE_H_

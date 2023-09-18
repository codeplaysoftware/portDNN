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
#ifndef PORTDNN_TEST_ROI_ALIGN_EVENT_DEPENDENCIES_FIXTURE_H
#define PORTDNN_TEST_ROI_ALIGN_EVENT_DEPENDENCIES_FIXTURE_H

#include <gtest/gtest.h>

#include "portdnn/helpers/scope_exit.h"

#include "portdnn/roi_align/launch.h"
#include "portdnn/roi_align/operators.h"
#include "portdnn/roi_align/params.h"

#include "test/backend/backend_test_fixture.h"
#include "test/gen/iota_initialised_data.h"
#include "test/helpers/dependency_check.h"

#include <numeric>
#include <string>
#include <vector>

template <typename DType>
struct RoiAlignFixtureEventDependencies
    : public BackendTestFixture<sycldnn::backend::SNNUSMBackend> {
  using DataType = DType;
  using BatchIndicesType = int32_t;
  using IndexType = int32_t;

  void test_roi_align_event_dependencies(
      sycldnn::roi_align::RoiAlignParams const& params) {
    std::vector<DataType> rois(params.num_rois * params.roi_cols);
    std::iota(rois.begin(), rois.end(), DataType(0));
    std::vector<BatchIndicesType> batch_indices =
        iota_initialised_data(params.num_rois, 1);
    const std::vector<IndexType> X_shape = {params.batch, params.channels,
                                            params.in_width, params.in_height};
    const IndexType X_num_elems = std::accumulate(
        X_shape.begin(), X_shape.end(), 1, std::multiplies<IndexType>());
    std::vector<DataType> X(X_num_elems);
    std::iota(X.begin(), X.end(), DataType(0));

    size_t const out_num_elems = params.num_rois * params.channels *
                                 params.out_height * params.out_width;
    std::vector<DataType> out_data_max_pool(out_num_elems, DataType(0));
    std::vector<DataType> out_data_avg_pool(out_num_elems, DataType(0));

    auto& provider = this->provider_;
    auto& backend = provider.get_backend();

    {
      auto X_gpu = provider.get_initialised_device_memory(X.size(), X);
      auto rois_gpu = provider.get_initialised_device_memory(rois.size(), rois);
      auto batch_indices_gpu = provider.get_initialised_device_memory(
          batch_indices.size(), batch_indices);
      auto out_gpu_max_pool = provider.get_initialised_device_memory(
          out_num_elems, out_data_max_pool);
      auto out_gpu_avg_pool = provider.get_initialised_device_memory(
          out_num_elems, out_data_avg_pool);
      SNN_ON_SCOPE_EXIT {
        provider.deallocate_ptr(X_gpu);
        provider.deallocate_ptr(rois_gpu);
        provider.deallocate_ptr(batch_indices_gpu);
        provider.deallocate_ptr(out_gpu_max_pool);
        provider.deallocate_ptr(out_gpu_avg_pool);
      };

      try {
        dependency_test_params dep_test_params;
        cl::sycl::event dependee_e = create_event(backend, dep_test_params);

        // Max pooling
        auto status = sycldnn::roi_align::launch<DataType, BatchIndicesType,
                                                 sycldnn::roi_align::MaxPool>(
            X_gpu, rois_gpu, batch_indices_gpu, out_gpu_max_pool, params,
            backend, {dependee_e});

        check_dependency(dependee_e, status.event, backend, dep_test_params);

      } catch (cl::sycl::exception const& e) {
        throw std::runtime_error(e.what());
      }

      try {
        dependency_test_params dep_test_params;
        cl::sycl::event dependee_e = create_event(backend, dep_test_params);

        // Average pooling
        auto status =
            sycldnn::roi_align::launch<DataType, BatchIndicesType,
                                       sycldnn::roi_align::AveragePool>(
                X_gpu, rois_gpu, batch_indices_gpu, out_gpu_avg_pool, params,
                backend, {dependee_e});

        check_dependency(dependee_e, status.event, backend, dep_test_params);

      } catch (cl::sycl::exception const& e) {
        throw std::runtime_error(e.what());
      }
    }
  }
};

#endif  // PORTDNN_TEST_ROI_ALIGN_EVENT_DEPENDENCIES_FIXTURE_H

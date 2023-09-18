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

#ifndef PORTDNN_TEST_SCATTER_ND_FIXTURE_H_
#define PORTDNN_TEST_SCATTER_ND_FIXTURE_H_

#include <gtest/gtest.h>

#include "portdnn/backend/snn_usm_backend.h"
#include "portdnn/helpers/scope_exit.h"

#include "portdnn/scatter_nd/launch.h"
#include "portdnn/scatter_nd/operators.h"
#include "portdnn/scatter_nd/sizes.h"

#include "test/backend/backend_test_fixture.h"
#include "test/gen/iota_initialised_data.h"
#include "test/helpers/dependency_check.h"
#include "test/helpers/float_comparison.h"

inline sycldnn::scatter_nd::ScatterNDParams getScatterNDParams(
    std::array<int, 4> in_shape, std::array<int, 2> ind_shape) {
  sycldnn::scatter_nd::ScatterNDParams params;
  params.input_dims = std::vector<int>(in_shape.begin(), in_shape.end());
  params.index_dims = std::vector<int>(ind_shape.begin(), ind_shape.end());
  return params;
}

template <typename DType, typename IType, typename ScatterNDType>
struct ScatterNDEventFixture
    : public BackendTestFixture<sycldnn::backend::SNNUSMBackend> {
  using DataType = DType;
  using IndexType = IType;
  void test_scatter_nd(std::vector<DataType> const& input,
                       std::vector<IndexType> const& indices,
                       std::vector<DataType> const& updates,
                       sycldnn::scatter_nd::ScatterNDParams const& params) {
    auto sizes = get_sizes(params);
    auto input_size = sizes.output_size;
    auto num_updates = sizes.num_updates;
    auto index_depth = sizes.index_depth;
    auto indices_size = num_updates * index_depth;
    auto updates_size = num_updates * sizes.slice_size;

    std::vector<DataType> output(input_size);

    auto& provider = this->provider_;
    auto& backend = provider.get_backend();

    auto inp_gpu = provider.get_initialised_device_memory(input_size, input);
    auto ind_gpu =
        provider.get_initialised_device_memory(indices_size, indices);
    auto upd_gpu =
        provider.get_initialised_device_memory(updates_size, updates);
    auto out_gpu = provider.get_initialised_device_memory(input_size, output);

    SNN_ON_SCOPE_EXIT {
      provider.deallocate_ptr(inp_gpu);
      provider.deallocate_ptr(ind_gpu);
      provider.deallocate_ptr(upd_gpu);
      provider.deallocate_ptr(out_gpu);
    };

    dependency_test_params dep_test_params;
    cl::sycl::event dependee_e = create_event(backend, dep_test_params);

    auto status =
        sycldnn::scatter_nd::launch<DataType, IndexType, ScatterNDType>(
            inp_gpu, ind_gpu, upd_gpu, out_gpu, params, backend,
            std::vector<cl::sycl::event>{dependee_e});

    ASSERT_EQ(sycldnn::StatusCode::OK, status.status);
    check_dependency(dependee_e, status.event, backend, dep_test_params);
  }
};

#endif  // PORTDNN_TEST_SCATTER_ND_FIXTURE_H_

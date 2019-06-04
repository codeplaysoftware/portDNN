/*
 * Copyright 2019 Codeplay Software Ltd
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
#ifndef SYCLDNN_SRC_POOLING_QUEUE_MAX_GRAD_KERNEL_IMPL_H_
#define SYCLDNN_SRC_POOLING_QUEUE_MAX_GRAD_KERNEL_IMPL_H_

#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include "sycldnn/pooling/params.h"

#include "src/pooling/kernels.h"

#include <CL/sycl.hpp>

namespace sycldnn {
namespace pooling {
namespace internal {

template <typename T, typename Index, template <typename U> class PoolType,
          typename Direction, int VectorWidth, bool UseFastDiv>
SNNStatus queue_max_grad_pooling(BaseMemObject<T const>& input_mem,
                                 BaseMemObject<T const>& output_mem,
                                 BaseMemObject<T const>& input_backprop_mem,
                                 BaseMemObject<T>& output_backprop_mem,
                                 const PoolingParams& pp, size_t threads,
                                 cl::sycl::queue& queue) {
  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    auto input_data = input_mem.read_accessor(cgh);
    auto output_data = output_mem.read_accessor(cgh);
    auto input_backprop = input_backprop_mem.read_accessor(cgh);
    auto output_backprop = output_backprop_mem.write_accessor(cgh);

    PoolingOp<T, Index, PoolType, Direction, VectorWidth, UseFastDiv> pool{
        input_data, output_data, input_backprop, output_backprop, pp};

    cgh.parallel_for(cl::sycl::range<1>{threads}, pool);
  });

  return {event, StatusCode::OK};
}

}  // namespace internal
}  // namespace pooling
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_POOLING_QUEUE_MAX_GRAD_KERNEL_IMPL_H_

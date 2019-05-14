/*
 * Copyright 2018 Codeplay Software Ltd
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

#ifndef SYCLDNN_INCLUDE_POOLING_LAUNCH_INTERNAL_H_
#define SYCLDNN_INCLUDE_POOLING_LAUNCH_INTERNAL_H_

#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include "sycldnn/pooling/operators.h"
#include "sycldnn/pooling/params.h"

namespace sycldnn {
namespace pooling {
namespace internal {

template <typename T, template <typename> class PoolType>
struct IsAverage {
  static constexpr bool value = std::is_same<PoolType<T>, Average<T>>::value;
};

template <typename T, template <typename> class PoolType, typename Direction>
struct IsAverageGradient {
  static constexpr bool value = IsAverage<T, PoolType>::value &&
                                std::is_same<Direction, Backpropagate>::value;
};

template <typename T, template <typename> class PoolType>
struct IsMax {
  static constexpr bool value = std::is_same<PoolType<T>, Max<T>>::value ||
                                std::is_same<PoolType<T>, MaxWithNan<T>>::value;
};

template <typename T, template <typename> class PoolType, typename Direction>
struct IsMaxGradient {
  static constexpr bool value = IsMax<T, PoolType>::value &&
                                std::is_same<Direction, Backpropagate>::value;
};

template <typename T, template <typename> class PoolType, typename Direction>
using DisableIfMaxGradient =
    typename std::enable_if<!IsMaxGradient<T, PoolType, Direction>::value,
                            int>::type;

template <typename T, template <typename> class PoolType, typename Direction>
using EnableIfMaxGradient =
    typename std::enable_if<IsMaxGradient<T, PoolType, Direction>::value,
                            int>::type;

template <typename T, template <typename> class PoolType, typename Direction,
          DisableIfMaxGradient<T, PoolType, Direction> = 0>
SNNStatus launch_pooling(BaseMemObject<T const>& input,
                         BaseMemObject<T>& output, const PoolingParams& pp,
                         cl::sycl::queue& queue);

template <typename T, template <typename> class PoolType, typename Direction,
          EnableIfMaxGradient<T, PoolType, Direction> = 0>
SNNStatus launch_pooling(BaseMemObject<T const>& inp_data,
                         BaseMemObject<T const>& outp_data,
                         BaseMemObject<T const>& inp_backprop,
                         BaseMemObject<T>& outp_backprop,
                         const PoolingParams& pp, cl::sycl::queue& queue);

}  // namespace internal
}  // namespace pooling
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_POOLING_LAUNCH_INTERNAL_H_

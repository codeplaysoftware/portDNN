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

#ifndef PORTDNN_INCLUDE_POOLING_LAUNCH_INTERNAL_H_
#define PORTDNN_INCLUDE_POOLING_LAUNCH_INTERNAL_H_

#include "portdnn/mem_object.h"
#include "portdnn/status.h"

#include "portdnn/pooling/operators.h"
#include "portdnn/pooling/params.h"

#include "portdnn/export.h"

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
          template <typename> class MemObj,
          DisableIfMaxGradient<T, PoolType, Direction> = 0>
SNN_EXPORT SNNStatus launch_pooling(
    MemObj<T const>& input, MemObj<T>& output, const PoolingParams& pp,
    cl::sycl::queue& queue, const std::vector<cl::sycl::event>& events = {});

template <typename T, template <typename> class PoolType, typename Direction,
          template <typename> class MemObj,
          EnableIfMaxGradient<T, PoolType, Direction> = 0>
SNN_EXPORT SNNStatus
launch_pooling(MemObj<T const>& inp_data, MemObj<T const>& outp_data,
               MemObj<T const>& inp_backprop, MemObj<T>& outp_backprop,
               const PoolingParams& pp, cl::sycl::queue& queue,
               const std::vector<cl::sycl::event>& events = {});

template <typename T, template <typename> class PoolType, typename Direction,
          typename Backend, DisableIfMaxGradient<T, PoolType, Direction> = 0>
SNN_EXPORT SNNStatus sublaunch(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T> output, const PoolingParams& pp,
    Backend& backend, const std::vector<cl::sycl::event>& events = {}) {
  auto sizes = get_sizes<Direction>(pp);

  auto inp_mem = backend.get_mem_object(input, sizes.input_size);
  auto outp_mem = backend.get_mem_object(output, sizes.output_size);

  auto queue = backend.get_queue();
  return internal::launch_pooling<T, PoolType, Direction>(inp_mem, outp_mem, pp,
                                                          queue, events);
}

template <typename T, template <typename> class PoolType, typename Direction,
          typename Backend, EnableIfMaxGradient<T, PoolType, Direction> = 0>
SNN_EXPORT SNNStatus
sublaunch(typename Backend::template pointer_type<T const> inp_data,
          typename Backend::template pointer_type<T const> outp_data,
          typename Backend::template pointer_type<T const> inp_backprop,
          typename Backend::template pointer_type<T> outp_backprop,
          const PoolingParams& pp, Backend& backend,
          const std::vector<cl::sycl::event>& events = {}) {
  auto fwd_sizes = get_sizes<Forward>(pp);
  auto back_sizes = get_sizes<Backpropagate>(pp);

  auto inp_data_access = backend.get_mem_object(inp_data, fwd_sizes.input_size);
  auto outp_data_access =
      backend.get_mem_object(outp_data, fwd_sizes.output_size);
  auto inp_backprop_access =
      backend.get_mem_object(inp_backprop, back_sizes.input_size);
  auto outp_backprop_access =
      backend.get_mem_object(outp_backprop, back_sizes.output_size);

  auto queue = backend.get_queue();
  return internal::launch_pooling<T, PoolType, Direction>(
      inp_data_access, outp_data_access, inp_backprop_access,
      outp_backprop_access, pp, queue, events);
}

}  // namespace internal
}  // namespace pooling
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_POOLING_LAUNCH_INTERNAL_H_

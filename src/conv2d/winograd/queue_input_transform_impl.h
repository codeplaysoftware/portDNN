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
#ifndef PORTDNN_SRC_CONV2D_WINOGRAD_QUEUE_INPUT_TRANSFORM_IMPL_H_
#define PORTDNN_SRC_CONV2D_WINOGRAD_QUEUE_INPUT_TRANSFORM_IMPL_H_

#include "portdnn/mem_object.h"

#include "src/conv2d/winograd/queue_input_transform.h"

#include "src/conv2d/winograd/kernels/extract_input_transform.h"

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace winograd {

namespace {

/** Round up a value to the nearest multiple of 4. */
auto round_up = [](int val) {
  constexpr int pow_two_multiple = 4;
  return helpers::round_up_to_nearest_multiple(val, pow_two_multiple);
};

/** Get the number of threads for a given convolution. */
inline cl::sycl::range<1> get_thread_range(Conv2DParams const& params,
                                           TileInfo const& tile_info,
                                           int channel_vector) {
  size_t n_threads = round_up(params.batch * tile_info.rows * tile_info.cols *
                              params.channels / channel_vector);
  return cl::sycl::range<1>{n_threads};
}

}  // namespace

template <typename T, typename Index, typename ConvType, int M, int N, int R,
          int S, int ChannelVector, template <typename> class MemObj>
SNNStatus queue_input_transform(MemObj<T const>& input_mem,
                                MemObj<T>& transform_mem,
                                Conv2DParams const& params,
                                TileInfo const& tile_info,
                                cl::sycl::queue& queue,
                                const std::vector<cl::sycl::event>& events) {
  using Functor = ExtractInputTiles<T, Index, ChannelVector, M, N, R, S,
                                    ConvType, is_usm_obj_v<MemObj<T>, T>>;

  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    cgh.depends_on(events);
    auto input = input_mem.read_mem(cgh);
    auto transform = transform_mem.write_mem(cgh);
    auto range = get_thread_range(params, tile_info, ChannelVector);
    Functor conv{params, tile_info, input, transform};

    cgh.parallel_for(range, conv);
  });
  return SNNStatus{event, StatusCode::OK};
}

}  // namespace winograd
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_SRC_CONV2D_WINOGRAD_QUEUE_INPUT_TRANSFORM_H_

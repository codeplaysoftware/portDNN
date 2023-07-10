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
#ifndef SYCLDNN_SRC_CONV2D_IM2COL_QUEUE_INPUT_TRANSFORM_IMPL_H_
#define SYCLDNN_SRC_CONV2D_IM2COL_QUEUE_INPUT_TRANSFORM_IMPL_H_

#include "sycldnn/format_type.h"
#include "sycldnn/mem_object.h"

#include "sycldnn/conv2d/params.h"

#include "sycldnn/helpers/ratio.h"

#include "src/conv2d/im2col/kernels/extract_input_tiles.h"
#include "src/conv2d/im2col/queue_input_transform.h"

#include <CL/sycl.hpp>

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace im2col {

namespace {

/** Round up a value to the nearest multiple of 4. */
auto round_up = [](int val) {
  constexpr int pow_two_multiple = 4;
  return helpers::round_up_to_nearest_multiple(val, pow_two_multiple);
};

template <int VectorWidth, typename ConvType,
          typename std::enable_if<
              !std::is_same<ConvType, conv_type::InputBackprop>::value,
              int>::type = 0>
cl::sycl::range<3> get_thread_range(Conv2DParams const& params) {
  size_t x = 0;
  size_t y = 0;
  size_t z = 0;
  if (params.input_format == DataFormat::NCHW) {
    x = round_up(params.channels);
    y = round_up(params.in_cols / VectorWidth);
  } else if (params.input_format == DataFormat::NHWC) {
    x = round_up(params.channels / VectorWidth);
    y = round_up(params.in_cols);
  }
  z = round_up(params.in_rows * params.batch);
  return cl::sycl::range<3>{x, y, z};
}

template <
    int VectorWidth, typename ConvType,
    typename std::enable_if<
        std::is_same<ConvType, conv_type::InputBackprop>::value, int>::type = 0>
cl::sycl::range<3> get_thread_range(Conv2DParams const& params) {
  size_t x = round_up(params.features / VectorWidth);
  size_t y = round_up(params.out_cols);
  size_t z = round_up(params.out_rows * params.batch);
  return cl::sycl::range<3>{x, y, z};
}

}  // namespace

template <typename T, typename Index, int VectorWidth, typename ConvType,
          typename Layout, template <typename> class MemObj>
SNNStatus queue_input_transform(MemObj<T const>& input_mem,
                                MemObj<T>& output_mem,
                                Conv2DParams const& params, int tile_size,
                                cl::sycl::queue& queue,
                                const std::vector<cl::sycl::event>& events) {
  constexpr bool is_usm = is_usm_obj_v<MemObj<T>, T>;

  using Functor =
      ExtractInputTiles<T, Index, VectorWidth, ConvType, is_usm, Layout>;

  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    cgh.depends_on(events);
    auto input = input_mem.read_mem(cgh);
    auto output = output_mem.write_mem(cgh);
    auto range = get_thread_range<VectorWidth, ConvType>(params);
    Functor conv{tile_size, params, input, output};

    cgh.parallel_for(range, conv);
  });
  return SNNStatus{event, StatusCode::OK};
}

}  // namespace im2col
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_CONV2D_IM2COL_QUEUE_INPUT_TRANSFORM_IMPL_H_

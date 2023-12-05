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
#ifndef PORTDNN_INCLUDE_INTERNAL_CONV2D_WINOGRAD_LAUNCH_FILTER_TRANSFORM_H_
#define PORTDNN_INCLUDE_INTERNAL_CONV2D_WINOGRAD_LAUNCH_FILTER_TRANSFORM_H_

#include "portdnn/mem_object.h"
#include "portdnn/status.h"

#include "portdnn/conv2d/conv_type.h"
#include "portdnn/conv2d/params.h"

#include "portdnn/internal/conv2d/winograd/tile_info.h"

#include <stddef.h>

#include <CL/sycl.hpp>

#include "portdnn/export.h"

/**
 * \file
 * Contains the sycldnn::conv2d::internal::winograd::launch_filter_transform()
 * function to launch the kernel to write the Winograd filter transform to a
 * temporary buffer.
 */

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace winograd {

/**
 * Launch the Winograd filter transform kernel.
 *
 * Will compute the Winograd transform for the input filter tensor, writing the
 * result into the output tensor.
 *
 * \param input     Input filter tensor
 * \param transform Output temporary transform tensor
 * \param params    Kernel parameters for the convolution
 * \param tile_info Winograd tile information
 * \param queue     SYCL queue to enqueue the kernels to
 * \param events    Vector of events to synchronize on before launching kernel
 * \return An SNNStatus event containing an event corresponding to the last
 * kernel launched.
 */
template <typename T, typename ConvType, int M, int N, int R, int S,
          template <typename> class MemObj>
SNN_EXPORT SNNStatus launch_filter_transform(
    MemObj<T const>& input, MemObj<T>& transform, Conv2DParams const& params,
    TileInfo const& tile_info, cl::sycl::queue& queue,
    const std::vector<cl::sycl::event>& events);

/**
 * Extract the buffers from the backend and launch the Winograd filter transform
 * kernel.
 *
 * \param filter    Input filter tensor
 * \param transform Output temporary transform tensor
 * \param params    Kernel parameters for the convolution
 * \param tile_info Winograd tile information
 * \param backend   Backend to provide SYCL buffers from the pointers
 * \param events    Vector of events to synchronize on before launching kernel
 * \return An SNNStatus event containing an event corresponding to the last
 * kernel launched.
 */
template <typename T, typename ConvType, int M, int N, int R, int S,
          typename Backend>
SNNStatus launch_filter_transform(
    typename Backend::template internal_pointer_type<T const> filter,
    typename Backend::template internal_pointer_type<T> transform,
    Conv2DParams const& params, TileInfo const& tile_info, Backend& backend,
    const std::vector<cl::sycl::event>& events) {
  constexpr int A = M + R - 1;
  constexpr int B = N + S - 1;

  size_t const filter_size = R * S * params.channels * params.features;
  auto filter_acc = backend.get_mem_object_internal(filter, filter_size);

  size_t const transform_size = A * B * params.channels * params.features;
  auto transform_acc =
      backend.get_mem_object_internal(transform, transform_size);

  cl::sycl::queue queue = backend.get_queue();
  return launch_filter_transform<T, ConvType, M, N, R, S>(
      filter_acc, transform_acc, params, tile_info, queue, events);
}

/**
 * Extract the buffers from the backend and launch the Winograd filter transform
 * kernel for a filter backprop convolution
 *
 * \param filter    Input filter tensor
 * \param transform Output temporary transform tensor
 * \param params    Kernel parameters for the convolution
 * \param tile_info Winograd tile information
 * \param backend   Backend to provide SYCL buffers from the pointers
 * \param events    Vector of events to synchronize on before launching kernel
 * \return An SNNStatus event containing an event corresponding to the last
 * kernel launched.
 */
template <typename T, int M, int N, int R, int S, typename Backend>
SNNStatus launch_filter_transform_filter_backprop(
    typename Backend::template internal_pointer_type<T const> filter,
    typename Backend::template internal_pointer_type<T> transform,
    Conv2DParams const& params, TileInfo const& tile_info, Backend& backend,
    const std::vector<cl::sycl::event>& events) {
  using ConvType = conv_type::FilterBackprop;
  constexpr int A = M + R - 1;
  constexpr int B = N + S - 1;

  size_t const filter_size =
      params.batch * params.window_rows * params.window_cols * params.features;
  auto filter_acc = backend.get_mem_object_internal(filter, filter_size);

  size_t const transform_size =
      A * B * params.batch * tile_info.number * params.features;
  auto transform_acc =
      backend.get_mem_object_internal(transform, transform_size);

  cl::sycl::queue queue = backend.get_queue();
  return launch_filter_transform<T, ConvType, M, N, R, S>(
      filter_acc, transform_acc, params, tile_info, queue, events);
}

}  // namespace winograd
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_INTERNAL_CONV2D_WINOGRAD_LAUNCH_FILTER_TRANSFORM_H_
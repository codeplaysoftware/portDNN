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
#ifndef SYCLDNN_INCLUDE_INTERNAL_CONV2D_WINOGRAD_LAUNCH_OUTPUT_TRANSFORM_H_
#define SYCLDNN_INCLUDE_INTERNAL_CONV2D_WINOGRAD_LAUNCH_OUTPUT_TRANSFORM_H_

#include "sycldnn/accessor_types.h"
#include "sycldnn/status.h"

#include "sycldnn/conv2d/params.h"
#include "sycldnn/internal/conv2d/winograd/tile_info.h"

#include <CL/sycl.hpp>

/**
 * \file
 * Contains the sycldnn::conv2d::internal::winograd::launch_output_transform()
 * function to launch the kernel to write the Winograd output transform to the
 * user provided output buffer.
 */
namespace sycldnn {
namespace conv2d {
namespace internal {
namespace winograd {

/**
 * Launch the Winograd output transform kernel.
 *
 * Will compute the Winograd transform converting the intermediate tensor to the
 * convolution output, writing the
 * result into the output tensor.
 *
 * \param intermediate Intermediate tensor
 * \param output       Output temporary transform tensor
 * \param params       Kernel parameters for the convolution
 * \param tile_info    Winograd tile information
 * \param queue        SYCL queue to enqueue the kernels to
 * \return An SNNStatus event containing an event corresponding to the last
 * kernel launched.
 */
template <typename T, typename ConvType, int M, int N, int R, int S,
          bool Accumulate = false>
SNNStatus launch_output_transform(ReadAccessor<T const> intermediate,
                                  WriteAccessor<T> output,
                                  Conv2DParams const& params,
                                  TileInfo const& tile_info,
                                  cl::sycl::queue& queue);

/**
 * Extract the buffers from the backend and launch the Winograd output transform
 * kernel.
 *
 * \param inter     Intermediate tensor
 * \param output    Output temporary transform tensor
 * \param params    Kernel parameters for the convolution
 * \param tile_info Winograd tile information
 * \param backend   Backend to provide SYCL buffers from the pointers
 * \return An SNNStatus event containing an event corresponding to the last
 * kernel launched.
 */
template <typename T, typename ConvType, int M, int N, int R, int S,
          typename Backend>
SNNStatus launch_output_transform(
    typename Backend::template internal_pointer_type<T const> inter,
    typename Backend::template internal_pointer_type<T> output,
    Conv2DParams const& params, TileInfo const& tile_info, Backend& backend) {
  constexpr int A = M + R - 1;
  constexpr int B = N + S - 1;

  size_t const inter_size =
      A * B * params.batch * tile_info.number * params.features;
  auto inter_buffer = backend.get_buffer_internal(inter, inter_size);
  size_t const inter_offset = backend.get_offset_internal(inter);
  ReadAccessor<T const> inter_acc{inter_buffer, cl::sycl::range<1>{inter_size},
                                  cl::sycl::id<1>{inter_offset}};

  size_t const output_size =
      params.batch * params.out_rows * params.out_cols * params.features;
  auto output_buffer = backend.get_buffer_internal(output, output_size);
  size_t const output_offset = backend.get_offset_internal(output);
  WriteAccessor<T> output_acc{output_buffer, cl::sycl::range<1>{output_size},
                              cl::sycl::id<1>{output_offset}};

  cl::sycl::queue queue = backend.get_queue();
  return launch_output_transform<T, ConvType, M, N, R, S>(
      inter_acc, output_acc, params, tile_info, queue);
}

/**
 * Extract the buffers from the backend and launch the Winograd output transform
 * kernel for a filter backprop convolution.
 *
 * \param inter     Intermediate tensor
 * \param output    Output temporary transform tensor
 * \param params    Kernel parameters for the convolution
 * \param tile_info Winograd tile information
 * \param backend   Backend to provide SYCL buffers from the pointers
 * \return An SNNStatus event containing an event corresponding to the last
 * kernel launched.
 */
template <typename T, int M, int N, int R, int S, bool Accumulate,
          typename Backend>
SNNStatus launch_output_transform_filter_backprop(
    typename Backend::template internal_pointer_type<T const> inter,
    typename Backend::template internal_pointer_type<T> output,
    Conv2DParams const& params, TileInfo const& tile_info, Backend& backend) {
  using ConvType = conv_type::FilterBackprop;
  constexpr int A = M + R - 1;
  constexpr int B = N + S - 1;

  size_t const inter_size =
      A * B * params.batch * tile_info.number * params.features;
  auto inter_buffer = backend.get_buffer_internal(inter, inter_size);
  size_t const inter_offset = backend.get_offset_internal(inter);
  ReadAccessor<T const> inter_acc{inter_buffer, cl::sycl::range<1>{inter_size},
                                  cl::sycl::id<1>{inter_offset}};

  size_t const output_size = M * N * params.channels * params.features;
  auto output_buffer = backend.get_buffer_internal(output, output_size);
  size_t const output_offset = backend.get_offset_internal(output);
  WriteAccessor<T> output_acc{output_buffer, cl::sycl::range<1>{output_size},
                              cl::sycl::id<1>{output_offset}};

  cl::sycl::queue queue = backend.get_queue();
  return launch_output_transform<T, ConvType, M, N, R, S, Accumulate>(
      inter_acc, output_acc, params, tile_info, queue);
}

}  // namespace winograd
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_INTERNAL_CONV2D_WINOGRAD_LAUNCH_OUTPUT_TRANSFORM_H

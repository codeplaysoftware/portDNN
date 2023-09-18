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
#ifndef PORTDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_H_
#define PORTDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_H_

#include "portdnn/conv2d/params.h"
#include "portdnn/helpers/macros.h"
#include "portdnn/status.h"

#include "portdnn/internal/conv2d/alloc_info.h"
#include "portdnn/internal/conv2d/batch_info.h"
#include "portdnn/internal/conv2d/internal_pointer_set.h"

#include "portdnn/internal/conv2d/im2col/allocated_pointer_set.h"
#include "portdnn/internal/conv2d/im2col/full_pointer_set.h"
#include "portdnn/internal/conv2d/im2col/kernel_params.h"
#include "portdnn/internal/conv2d/im2col/launch_filter_transform.h"
#include "portdnn/internal/conv2d/im2col/launch_input_transform.h"
#include "portdnn/internal/conv2d/im2col/offsets.h"
#include "portdnn/internal/conv2d/im2col/tile_info.h"
#include "portdnn/internal/conv2d/im2col/transform_sizes.h"
#include "portdnn/internal/conv2d/im2col/workspace_pointer_set.h"
#include "portdnn/internal/transpose/launch.h"
namespace sycldnn {
namespace conv2d {
namespace internal {
namespace im2col {

/** Launch the input transform and matmul to compute im2col. */
template <typename T, typename ConvType, typename Backend,
          typename std::enable_if<
              !std::is_same<ConvType, conv_type::FilterBackprop>::value,
              int>::type = 0>
static SNNStatus launch_im2col_for_minibatch(
    FullPointerSet<T, Backend, ConvType> const& pointers, size_t in_offset,
    size_t out_offset, TileInfo const& tile_info, Conv2DParams const& params,
    Backend& backend, const std::vector<cl::sycl::event>& events) {
  using ConstPointer =
      typename FullPointerSet<T, Backend, ConvType>::ConstPointer;

  const auto filter_size =
      std::is_same<ConvType, conv_type::InputBackprop>::value
          ? 0
          : filter_transform_size<ConvType>(params);
  auto status = launch_input_transform(pointers, in_offset, filter_size,
                                       tile_info, params, backend, events);
  if (status.status != StatusCode::OK) {
    return status;
  }

  std::vector<cl::sycl::event> dependencies{status.event};

  int matmul_size;
  if (std::is_same<ConvType, conv_type::InputBackprop>::value) {
    matmul_size = params.channels / params.groups;
  } else {
    matmul_size = params.features / params.groups;
  }

  int const n_tiles = params.batch * tile_info.number;
  int const tile_size = tile_info.size;

  cl::sycl::event event;
  if (params.groups == 1) {
    // Regular convolution, no filter/output transformations are needed.
    if (params.filter_format == sycldnn::FilterFormat::FHWC) {
      event = backend.template matmul<false, true>(
          ConstPointer{pointers.transform}, ConstPointer{pointers.filter},
          pointers.output + out_offset, static_cast<T>(0), n_tiles, tile_size,
          matmul_size, dependencies);
    } else {
      event = backend.template matmul<false, false>(
          ConstPointer{pointers.transform}, ConstPointer{pointers.filter},
          pointers.output + out_offset, static_cast<T>(0), n_tiles, tile_size,
          matmul_size, dependencies);
    }
  } else {
    // Group convolution cases
    if (params.group_format == sycldnn::BatchFormat::STRIDED) {
      auto matmul_offset = n_tiles * tile_size * params.groups;

      if (params.filter_format == sycldnn::FilterFormat::FHWC) {
        event = backend.template batch_matmul<false, true>(
            ConstPointer{pointers.transform}, ConstPointer{pointers.filter},
            pointers.transform + matmul_offset, params.groups, n_tiles,
            tile_size, matmul_size, params.group_format, dependencies);
      } else {
        event = backend.template batch_matmul<false, false>(
            ConstPointer{pointers.transform + filter_size},
            ConstPointer{pointers.transform},
            pointers.transform + filter_size + matmul_offset, params.groups,
            n_tiles, tile_size, matmul_size, params.group_format, dependencies);
      }

      // Transpose needed at the end to reshape the output from GNHWC to NHWGC
      size_t const trans_size = params.groups * n_tiles * matmul_size;

      auto in_mem_obj =
          backend
              .get_mem_object(pointers.transform + filter_size + matmul_offset,
                              trans_size)
              .as_const();
      auto out_mem_obj =
          backend.get_mem_object(pointers.output + out_offset, trans_size);

      const std::vector<int> GNHWC_TO_NHWGC = {1, 2, 0, 3};
      auto queue = backend.get_queue();
      auto status = sycldnn::transpose::internal::launch(
          in_mem_obj, out_mem_obj,
          {params.groups, params.batch, tile_info.number, matmul_size},
          GNHWC_TO_NHWGC, queue, {event});

      if (status.status != StatusCode::OK) {
        return status;
      }

      event = status.event;
    } else {
      // Interleaved group format case. No filter/output transpose is needed.
      event = backend.template batch_matmul<false, false>(
          ConstPointer{pointers.transform}, ConstPointer{pointers.filter},
          pointers.output + out_offset, params.groups, n_tiles, tile_size,
          matmul_size, params.group_format, dependencies);
    }
  }
  return {event, StatusCode::OK};
}

/**
 * Launch the input transform and matmul to compute im2col for the filter
 * backprop pass.
 */
template <typename T, typename ConvType, typename Backend,
          typename std::enable_if<
              std::is_same<ConvType, conv_type::FilterBackprop>::value,
              int>::type = 0>
static SNNStatus launch_im2col_for_minibatch(
    FullPointerSet<T, Backend, ConvType> const& pointers, size_t in_offset,
    size_t out_offset, TileInfo const& tile_info, Conv2DParams const& params,
    Backend& backend, const std::vector<cl::sycl::event>& events) {
  using ConstPointer =
      typename FullPointerSet<T, Backend, ConvType>::ConstPointer;
  auto status = launch_input_transform(pointers, in_offset, 0, tile_info,
                                       params, backend, events);
  if (status.status != StatusCode::OK) {
    return status;
  }

  std::vector<cl::sycl::event> dependencies{status.event};

  const int n_tiles = tile_info.number;
  const int tile_size = params.batch * tile_info.size;

  cl::sycl::event matmul_event;
  if (in_offset == 0) {
    matmul_event = backend.template matmul<false, false>(
        ConstPointer{pointers.transform}, pointers.filter + out_offset,
        pointers.output, static_cast<T>(0), n_tiles, tile_size, params.features,
        dependencies);
  } else {
    matmul_event = backend.template matmul<false, false>(
        ConstPointer{pointers.transform}, pointers.filter + out_offset,
        pointers.output, static_cast<T>(1), n_tiles, tile_size, params.features,
        dependencies);
  }
  return {matmul_event, StatusCode::OK};
}

/** Loop over the minibatches to compute im2col. */
template <typename T, typename ConvType, typename Backend>
static SNNStatus launch_im2col_for_all_minibatches(
    FullPointerSet<T, Backend, ConvType> const& pointers,
    TileInfo const& tile_info, BatchInfo const& batch_info,
    Conv2DParams const& params, Backend& backend,
    const std::vector<cl::sycl::event>& events) {
  auto filter_status =
      launch_filter_transform(pointers, params, backend, events);
  if (filter_status.status != StatusCode::OK) {
    return filter_status;
  }

  auto kernel_params = get_kernel_params<ConvType>(params);
  kernel_params.batch = batch_info.images_per_batch;

  cl::sycl::event dep_event = filter_status.event;
  for (size_t i = 0; i < batch_info.n_batches; ++i) {
    auto offset =
        calculate_offsets<ConvType>(i, batch_info.images_per_batch, params);
    if (i == batch_info.n_batches - 1) {
      kernel_params.batch = batch_info.last_batch_size;
    }
    auto status =
        launch_im2col_for_minibatch(pointers, offset.in, offset.out, tile_info,
                                    kernel_params, backend, {dep_event});
    // Each minibatch depends on previous for safe re-use of transform buffer
    dep_event = status.event;
    if (status.status != StatusCode::OK) {
      return status;
    }
  }

  return SNNStatus{dep_event, StatusCode::OK};
}

/**
 * Split the input tensor into minibatches to ensure that the temporary
 * transform buffer can be safely allocated and create SYCL buffers using the
 * Backend. Then use im2col to compute the convolution for each minibatch.
 */
template <typename T, typename ConvType, typename Backend>
SNNStatus allocate_and_launch_im2col(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T const> filter,
    typename Backend::template pointer_type<T> output,
    Conv2DParams const& params, Backend& backend,
    const std::vector<cl::sycl::event>& events) {
  InternalPointerSet<T, Backend> pointers{input, filter, output, backend};

  auto const tile_info = im2col::get_tile_info<ConvType>(params);
  size_t const size_per_image =
      params.groups * tile_info.number * tile_info.size;
  im2col::AllocatedPointerSet<T, Backend, ConvType> all_pointers{
      pointers, size_per_image, params, backend};

  auto const batch_info = get_batch_info(all_pointers.allocated_transform_size,
                                         params.batch, size_per_image);

  const auto launch_status = im2col::launch_im2col_for_all_minibatches(
      all_pointers.to_full_pointer_set(), tile_info, batch_info, params,
      backend, events);
  all_pointers.pass_event_to_ptrs(launch_status.event);
  return launch_status;
}

/**
 * Use the provided workspace for the transform data. As the user may provide a
 * smaller workspace than is ideal, check the maximum size of the minibatch to
 * use, and split up the workspace as required. Then use im2col to compute the
 * convolution for each minibatch.
 */
template <typename T, typename ConvType, typename Backend>
SNNStatus launch_im2col_with_workspace(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T const> filter,
    typename Backend::template pointer_type<T> output,
    typename Backend::template pointer_type<T> workspace,
    Conv2DParams const& params, size_t workspace_size, Backend& backend,
    const std::vector<cl::sycl::event>& events) {
  InternalPointerSet<T, Backend> pointers{input, filter, output, backend};

  auto const tile_info = im2col::get_tile_info<ConvType>(params);
  size_t const size_per_image =
      params.groups * tile_info.number * tile_info.size;
  im2col::WorkspacePointerSet<T, Backend, ConvType> all_pointers{
      pointers, workspace, size_per_image, params, workspace_size, backend};

  auto const batch_info =
      get_batch_info(all_pointers.minibatch_size, params.batch);

  return im2col::launch_im2col_for_all_minibatches(
      all_pointers.to_full_pointer_set(), tile_info, batch_info, params,
      backend, events);
}

}  // namespace im2col

/**
 * The internal im2col convolution launcher.
 *
 * Use im2col to compute a convolution, by transforming the input data then
 * computing a matrix multiply with the filter to give the output.
 */
template <typename T, typename ConvType, typename Backend>
SNNStatus launch_im2col(typename Backend::template pointer_type<T const> input,
                        typename Backend::template pointer_type<T const> filter,
                        typename Backend::template pointer_type<T> output,
                        typename Backend::template pointer_type<T> workspace,
                        Conv2DParams const& params, size_t workspace_size,
                        Backend& backend,
                        const std::vector<cl::sycl::event>& events) {
  if (sycldnn::backend::supports_interleaved_matmul<Backend>::value &&
      (params.groups == params.channels) &&
      (params.groups == params.features) &&
      params.group_format == sycldnn::BatchFormat::STRIDED &&
      params.filter_format == sycldnn::FilterFormat::HWCF) {
    /**
     * Degenerate case of depthwise convolution where the feature_multiplier==1.
     * In this case the input and filter dimensions become NHWG and HWG
     * respectively. Thus, the groups are interleaved into the data and an
     * interleaved batch_matmul can be used. This prevents us from having to do
     * a filter and output transpose.
     */

    Conv2DParams interleaved_params = params;
    interleaved_params.group_format = sycldnn::BatchFormat::INTERLEAVED;
    if (workspace_size == 0) {
      return im2col::allocate_and_launch_im2col<T, ConvType>(
          input, filter, output, interleaved_params, backend, events);
    } else {
      return im2col::launch_im2col_with_workspace<T, ConvType>(
          input, filter, output, workspace, interleaved_params, workspace_size,
          backend, events);
    }
  }

  if (workspace_size == 0) {
    return im2col::allocate_and_launch_im2col<T, ConvType>(
        input, filter, output, params, backend, events);
  } else {
    return im2col::launch_im2col_with_workspace<T, ConvType>(
        input, filter, output, workspace, params, workspace_size, backend,
        events);
  }
}
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_H_

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
#ifndef PORTDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_LAUNCH_FILTER_TRANSFORM_H_
#define PORTDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_LAUNCH_FILTER_TRANSFORM_H_

#include "portdnn/backend/backend_helpers.h"
#include "portdnn/helpers/event_handling.h"
#include "portdnn/mem_object.h"
#include "portdnn/status.h"

#include "portdnn/conv2d/params.h"

#include "portdnn/internal/conv2d/im2col/full_pointer_set.h"
#include "portdnn/internal/conv2d/im2col/transform_sizes.h"

#include "portdnn/export.h"

#include "portdnn/internal/transpose/launch.h"
namespace sycldnn {
namespace conv2d {
namespace internal {
namespace im2col {

/**
 * Launch the filter transform to mirror the filters for the input backprop.
 *
 * Implemented in the compiled portDNN library.
 *
 * \param [in]  input  User provided filter tensor
 * \param [out] output Filter transform tensor to fill with transformed filter
 *                     values
 * \param [in]  params Kernel parameters for the convolution
 * \param [in]  queue  SYCL queue to enqueue the kernel to
 * \return An SNNStatus with event linked to the kernel launch or an error code.
 */
template <typename T, template <typename> class MemObj>
SNN_EXPORT SNNStatus launch_filter_transform(
    MemObj<T const>& input, MemObj<T>& output, Conv2DParams const& params,
    cl::sycl::queue& queue, const std::vector<cl::sycl::event>& events);

/**
 * For forward and filter backprop the original filter is used,
 *  so just return.*/
template <typename T, typename ConvType, typename Backend,
          typename std::enable_if<
              !std::is_same<ConvType, conv_type::InputBackprop>::value,
              int>::type = 0>
static SNNStatus launch_filter_transform(
    FullPointerSet<T, Backend, ConvType> const& pointers,
    Conv2DParams const& params, Backend& backend,
    const std::vector<cl::sycl::event>& events) {
  auto queue = backend.get_queue();
  if (filter_transform_size<ConvType>(params) == 0)
    return {sycldnn::helpers::multi_event_to_one(events, queue),
            StatusCode::OK};

  SNN_VALIDATE_PARAM(
      params.group_format != sycldnn::BatchFormat::INTERLEAVED ||
          params.filter_format == sycldnn::FilterFormat::HWCF,
      "Interleaved group format is only supported for HWCF filter format.");

  int const features_per_group = params.features / params.groups;
  int const channels_per_group = params.channels / params.groups;
  int const total_size = params.window_rows * params.window_cols *
                         channels_per_group * params.features;
  auto in_mem_obj = backend.get_mem_object(pointers.filter, total_size);
  auto out_mem_obj = backend.get_mem_object(pointers.transform, total_size);
  const std::vector<int> HWCGF_TO_HWCFG = {3, 0, 1, 2, 4};
  return sycldnn::transpose::internal::launch(
      in_mem_obj, out_mem_obj,
      {params.window_rows, params.window_cols, channels_per_group,
       params.groups, features_per_group},
      HWCGF_TO_HWCFG, queue, events);
}

/**
 * For the input backprop the filter needs to be mirrored.
 *
 * The AllocatedPointerSet will already have a temporary filter transform
 * buffer for this mirrored filter, so fill this with the filter values.
 */
template <
    typename T, typename ConvType, typename Backend,
    typename std::enable_if<
        std::is_same<ConvType, conv_type::InputBackprop>::value, int>::type = 0>
static SNNStatus launch_filter_transform(
    FullPointerSet<T, Backend, ConvType> const& pointers,
    Conv2DParams const& params, Backend& backend,
    const std::vector<cl::sycl::event>& events) {
  size_t const filter_size = params.window_rows * params.window_cols *
                             params.channels * params.features;
  auto filter_access =
      backend.get_mem_object_internal(pointers.original_filter, filter_size);

  auto transform_access =
      backend.get_mem_object_internal(pointers.filter, filter_size);

  cl::sycl::queue queue = backend.get_queue();
  return launch_filter_transform(filter_access, transform_access, params, queue,
                                 events);
}

}  // namespace im2col
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_LAUNCH_FILTER_TRANSFORM_H_

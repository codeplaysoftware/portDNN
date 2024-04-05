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
#ifndef SYCLDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_LAUNCH_FILTER_TRANSFORM_H_
#define SYCLDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_LAUNCH_FILTER_TRANSFORM_H_

#include "sycldnn/backend/backend_helpers.h"
#include "sycldnn/helpers/event_handling.h"
#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include "sycldnn/conv2d/params.h"

#include "sycldnn/internal/conv2d/im2col/full_pointer_set.h"
#include "sycldnn/internal/conv2d/im2col/transform_sizes.h"
#include "sycldnn/internal/conv2d/im2col/transpose_helper.h"

#include "sycldnn/export.h"

#include "sycldnn/internal/transpose/launch.h"
namespace sycldnn {
namespace conv2d {
namespace internal {
namespace im2col {

/**
 * Launch the filter transform to mirror the filters for the input backprop.
 *
 * Implemented in the compiled SYCL-DNN library.
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

  // When transform size is zero the original filter is used.
  // This happens mainly in the forward case when params.filter_format=FCHW or
  // params.groups=1 and during filter backprop if params.input_format=NHWC
  if (filter_transform_size<ConvType>(params) == 0)
    return {sycldnn::helpers::multi_event_to_one(events, queue),
            StatusCode::OK};

  using TransposeHelper = Im2colTransposeHelper<T, ConvType>;
  if constexpr (std::is_same_v<ConvType, conv2d::conv_type::FilterBackprop>) {
    size_t const trans_size = filter_transform_size<ConvType>(params);
    auto in_mem_obj =
        backend.get_mem_object(pointers.filter, trans_size).as_const();
    auto out_mem_obj = backend.get_mem_object(pointers.transform, trans_size);

    return TransposeHelper::transpose_filter(in_mem_obj, out_mem_obj, params,
                                             queue, events);
  } else {
    int const channels_per_group = params.channels / params.groups;
    int const total_size = params.window_rows * params.window_cols *
                           channels_per_group * params.features;
    auto in_mem_obj = backend.get_mem_object(pointers.filter, total_size);
    auto out_mem_obj = backend.get_mem_object(pointers.transform, total_size);
    return TransposeHelper::transpose_filter(in_mem_obj, out_mem_obj, params,
                                             queue, events);
  }
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

#endif  // SYCLDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_LAUNCH_FILTER_TRANSFORM_H_

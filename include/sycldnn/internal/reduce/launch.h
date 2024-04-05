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
#ifndef SYCLDNN_INCLUDE_INTERNAL_REDUCE_LAUNCH_H_
#define SYCLDNN_INCLUDE_INTERNAL_REDUCE_LAUNCH_H_

#include <CL/sycl.hpp>
#include <algorithm>
#include <string>
#include <unordered_map>

#include "sycldnn/export.h"
#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include "sycldnn/reduce/helpers.h"
#include "sycldnn/reduce/operators.h"
#include "sycldnn/reduce/params.h"
#include "sycldnn/reduce/sizes.h"

#include "sycldnn/helpers/event_handling.h"

#include "sycldnn/internal/reduce/identities.h"
#include "sycldnn/internal/reduce/internal_params.h"

namespace sycldnn {
namespace reduce {
namespace internal {

/**
 * The internal reduce launcher.
 *
 * Implemented in the compiled SYCL DNN library.
 */
template <typename T, typename Op, template <typename> class MemObj>
SNN_EXPORT SNNStatus launch(MemObj<T const>& input, MemObj<T>& output,
                            int batches, int outer, int inner,
                            cl::sycl::queue& queue,
                            const std::vector<cl::sycl::event>& events);

/**
 * The internal Nd reduce launcher.
 *
 * Implemented in the compiled SYCL DNN library.
 */
template <typename T, typename Op, template <typename> class MemObj>
SNN_EXPORT SNNStatus launch(MemObj<T const>& input, MemObj<T>& output,
                            internal::ReduceParams const& params,
                            cl::sycl::queue& queue,
                            const std::vector<cl::sycl::event>& events);

/**
 * The internal reduce sublauncher.
 * Performs checks, and creates memory objects.
 *
 * Implemented in the compiled SYCL DNN library.
 */
template <typename T, typename Op, typename Backend>
SNNStatus sublaunch(typename Backend::template pointer_type<T const> input,
                    typename Backend::template pointer_type<T> output,
                    int batches, int outer, int inner, Backend& backend,
                    const std::vector<cl::sycl::event>& events) {
  static_assert(std::is_same<Op, reduce::Add>::value ||
                    std::is_same<Op, reduce::Mean>::value ||
                    std::is_same<Op, reduce::Max>::value ||
                    std::is_same<Op, reduce::Min>::value,
                "Invalid Reduction Type");
  SNN_VALIDATE_PARAM(batches > 0, "The number of batches must be positive.");
  SNN_VALIDATE_PARAM(outer > 0, "The value of outer must be positive.");
  SNN_VALIDATE_PARAM(inner > 0, "The value of inner must be positive.");

  size_t in_size = batches * outer * inner;
  size_t out_size = batches * inner;

  auto in_acc = backend.get_mem_object(input, in_size);
  auto out_acc = backend.get_mem_object(output, out_size);

  return launch<T, Op>(in_acc, out_acc, batches, outer, inner,
                       backend.get_queue(), events);
}

/**
 * The internal Nd reduce sublauncher.
 * Performs checks, chooses correct configuration of reduction creates memory
 * objects, submits required number of kernel launches
 *
 * Implemented in the compiled SYCL DNN library.
 */
template <typename T, typename Op, typename Backend>
SNNStatus sublaunch(typename Backend::template pointer_type<T const> input,
                    typename Backend::template pointer_type<T> output,
                    NDReduceParams const& params, Backend& backend,
                    typename Backend::template pointer_type<T> workspace,
                    size_t workspace_size,
                    const std::vector<cl::sycl::event>& events) {
  static_assert(std::is_same<Op, reduce::Add>::value ||
                    std::is_same<Op, reduce::Mean>::value ||
                    std::is_same<Op, reduce::Max>::value ||
                    std::is_same<Op, reduce::Min>::value,
                "Invalid Reduction Type");
  SNN_VALIDATE_PARAM(params.axes.size() > 0,
                     "There needs to be at least one reduction axis.");
  SNN_VALIDATE_PARAM(params.dims.size() > 0,
                     "There needs to be at least one input dimesion");
  for (auto& dim : params.dims) {
    SNN_VALIDATE_PARAM(dim > 0, "Dimension sizes need to be positive.");
  }
  for (auto& ax : params.axes) {
    SNN_VALIDATE_PARAM(ax >= 0 && ax < static_cast<int>(params.dims.size()),
                       "Axis dimension is out of tensor dimensions scope.");
  }
  bool supports_subgroup = backend.supports_subgroup();
  SNN_VALIDATE_PARAM(supports_subgroup, "Device does not support subgroups");

  auto const sizes = sycldnn::reduce::get_sizes(params);
  auto in_acc = backend.get_mem_object(input, sizes.input_size);
  auto out_acc = backend.get_mem_object(output, sizes.output_size);

  // Sort axes to pass to simplify_axes
  auto axes = params.axes;
  std::sort(axes.begin(), axes.end());
  auto axes_mask = simplify_axes(params.dims, axes);

  std::vector<int> new_dims;
  std::vector<int> new_axes;
  squeeze_dims_and_axes(params.dims, axes_mask, new_dims, new_axes);

  std::sort(new_axes.begin(), new_axes.end(), std::greater<int>());

  auto num_dims = new_dims.size();
  auto num_reductions = new_axes.size();
  auto out_size_vec = get_first_two_out_sizes(new_dims, new_axes);

  // Setup internal parameter struct
  internal::ReduceParams internal_params;
  internal_params.max_wg_size = backend.get_max_wg_size();
  internal_params.max_sg_size = backend.get_max_sg_size();
  auto queue = backend.get_queue();

  std::vector<cl::sycl::event> memset_events;
  if (sizes.reduce_size > 1 && params.prepare_output) {
    memset_events.push_back(backend.fill(
        out_acc, sizes.output_size, internal::Identity<T, Op>::value, events));
  } else {
    memset_events = events;
  }

  if (sizes.reduce_size == 1) {
    // No reduction is needed, just copy input into output
    auto no_op_event = backend.copy(in_acc, out_acc, sizes.output_size, events);
    return {no_op_event, StatusCode::OK};
  } else if (num_reductions == 1) {
    auto const dims_begin = new_dims.begin();
    auto const dims_end = new_dims.end();
    auto const axis = new_axes[0];

    // Setup internal params
    internal_params.batch_size = std::accumulate(dims_begin, dims_begin + axis,
                                                 1, std::multiplies<int>());
    internal_params.reduce_size = new_dims[axis];
    internal_params.stride_size = std::accumulate(
        dims_begin + axis + 1, dims_end, 1, std::multiplies<int>());

    // Launch kernel
    return launch<T, Op>(in_acc, out_acc, internal_params, queue,
                         memset_events);
  } else if ((num_dims - num_reductions) == 1) {
    // (num_dims-num_reductions) describes the number of dimensions being kept,
    // hence if num_reductions>=2 and at this point the dimensions and axes
    // being squeezed beforehand, the only way this should be possible is if the
    // reduction operation looks like [reduce,keep,reduce].
    SNN_ASSERT(num_dims == 3 && num_reductions == 2,
               "There may contain unsqueezed dimensions and axes");
    SNN_ASSERT(new_axes[1] == 0 && new_axes[0] == 2,
               "There is an issue with the squeeze_dims_and_axes()");

    // Setup internal params
    internal_params.batch_size = new_dims[0] * new_dims[1];
    internal_params.reduce_size = new_dims[2];
    internal_params.stride_size = 1;
    // Allows the kernel to reduce the outermost dim
    internal_params.batch_groups = new_dims[1];

    // Launch kernel
    return launch<T, Op>(in_acc, out_acc, internal_params, queue,
                         memset_events);
  } else {
    auto ref_workspace_size = sycldnn::reduce::get_workspace_size(params);
    SNN_VALIDATE_PARAM(ref_workspace_size <= workspace_size,
                       "Incorrect amount of workspace allocated");
    auto temp_mem1 = backend.get_mem_object(workspace, out_size_vec[0], 0);
    auto temp_mem2 =
        backend.get_mem_object(workspace, out_size_vec[1], out_size_vec[0]);
    if (params.prepare_output) {
      memset_events.push_back(backend.fill(temp_mem1, out_size_vec[0],
                                           internal::Identity<T, Op>::value,
                                           events));

      memset_events.push_back(backend.fill(temp_mem2, out_size_vec[1],
                                           internal::Identity<T, Op>::value,
                                           events));
    }
    cl::sycl::event reduction_event =
        sycldnn::helpers::multi_event_to_one(memset_events, queue);
    SNNStatus reduction_status;

    for (size_t red_iter = 0; red_iter < num_reductions; red_iter++) {
      auto mem_in = red_iter == 0 ? in_acc
                                  : red_iter % 2 != 0 ? temp_mem1.as_const()
                                                      : temp_mem2.as_const();
      auto mem_out = red_iter == num_reductions - 1
                         ? out_acc
                         : red_iter % 2 != 0 ? temp_mem2 : temp_mem1;
      auto const dims_begin = new_dims.begin();
      auto const dims_end = new_dims.end();
      auto const axis = new_axes[red_iter];
      internal_params.batch_size = std::accumulate(
          dims_begin, dims_begin + axis, 1, std::multiplies<int>());
      internal_params.reduce_size = new_dims[axis];
      if (axis < static_cast<int>(new_dims.size() - 1)) {
        internal_params.stride_size = std::accumulate(
            dims_begin + axis + 1, dims_end, 1, std::multiplies<int>());
      } else {
        internal_params.stride_size = 1;
      }
      // TODO Bug may fail if there are more than three non-contiguous
      // reduction dimensions Issue is that the output will have memory from
      // previous iterations in it.
      reduction_status = launch<T, Op>(mem_in, mem_out, internal_params, queue,
                                       {reduction_event});
      new_dims[new_axes[red_iter]] = 1;
      if (reduction_status.status != StatusCode::OK) {
        return reduction_status;
      }
      reduction_event = reduction_status.event;
    }
    return reduction_status;
  }
}

}  // namespace internal
}  // namespace reduce
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_INTERNAL_REDUCE_LAUNCH_H_

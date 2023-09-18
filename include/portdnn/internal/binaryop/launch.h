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

#ifndef PORTDNN_INCLUDE_BINARYOP_LAUNCH_INTERNAL_H_
#define PORTDNN_INCLUDE_BINARYOP_LAUNCH_INTERNAL_H_

#include <vector>

#include "portdnn/mem_object.h"
#include "portdnn/status.h"

#include "portdnn/binaryop/params.h"
#include "portdnn/helpers/dims.h"

#include "portdnn/backend/backend_helpers.h"

#include "portdnn/export.h"

namespace sycldnn {
namespace binaryop {
namespace internal {

/**
 * @brief Compute binary op out dimensions after performing a
 * multidirectional broadcast on input operands.
 *
 * @param[in] lhs_dims
 * @param[in] rhs_dims
 * @param[out] out_dims
 * @return Status code
 */
inline SNNStatus compute_out_dims(const std::vector<int>& lhs_dims,
                                  const std::vector<int>& rhs_dims,
                                  std::vector<int>& out_dims) {
  std::vector<int> smallest_dims =
      lhs_dims.size() <= rhs_dims.size() ? lhs_dims : rhs_dims;
  const std::vector<int>& largest_dims =
      lhs_dims.size() <= rhs_dims.size() ? rhs_dims : lhs_dims;

  // Prepend smallest_dims with 1s to match the number of dimensions
  size_t num_dims = largest_dims.size();
  while (smallest_dims.size() < num_dims) {
    smallest_dims.insert(smallest_dims.begin(), 1);
  }

  for (size_t i = 0; i < num_dims; ++i) {
    SNN_VALIDATE_PARAM(smallest_dims[i] == largest_dims[i] ||
                           smallest_dims[i] == 1 || largest_dims[i] == 1,
                       "Dimensions cannot be broadcasted.");
    out_dims.push_back(std::max(smallest_dims[i], largest_dims[i]));
  }
  return StatusCode::OK;
}

template <typename Op, typename T, template <typename> class MemObj>
SNN_EXPORT SNNStatus
launch_binaryop(MemObj<T const>& lhs, MemObj<T const>& rhs, MemObj<T>& out,
                std::vector<int> lhs_dims, std::vector<int> rhs_dims,
                const std::vector<int>& out_dims, cl::sycl::queue& queue,
                const std::vector<cl::sycl::event>& events);

template <typename T, typename Op, typename Backend>
SNNStatus sublaunch(typename Backend::template pointer_type<T const> lhs,
                    typename Backend::template pointer_type<T const> rhs,
                    typename Backend::template pointer_type<T> out,
                    const BinaryParams& params, Backend& backend,
                    const std::vector<cl::sycl::event>& events) {
  auto lhs_dims = params.lhs_dims;
  auto rhs_dims = params.rhs_dims;
  SNN_VALIDATE_PARAM(
      lhs_dims.size() <= MAX_DIMS,
      "Left operand size exceeds the maximum number of dimensions");
  SNN_VALIDATE_PARAM(
      rhs_dims.size() <= MAX_DIMS,
      "Right operand size exceeds the maximum number of dimensions");

  // Empty dimensions may be used to represent scalars.
  if (lhs_dims.size() == 0) {
    lhs_dims.push_back(1);
  }
  if (rhs_dims.size() == 0) {
    rhs_dims.push_back(1);
  }

  size_t lhs_size = helpers::get_total_size(lhs_dims);
  size_t rhs_size = helpers::get_total_size(rhs_dims);
  SNN_VALIDATE_PARAM(lhs_size > 0, "Left operand size cannot be zero.");
  SNN_VALIDATE_PARAM(rhs_size > 0, "Right operand size cannot be zero.");

  std::vector<int> out_dims;
  auto status = internal::compute_out_dims(lhs_dims, rhs_dims, out_dims);
  if (status.status != StatusCode::OK) {
    return status;
  }
  size_t out_size = helpers::get_total_size(out_dims);

  auto lhs_mem = backend.get_mem_object(lhs, lhs_size);
  auto rhs_mem = backend.get_mem_object(rhs, rhs_size);
  auto out_mem = backend.get_mem_object(out, out_size);
  auto queue = backend.get_queue();
  return internal::launch_binaryop<Op>(lhs_mem, rhs_mem, out_mem, lhs_dims,
                                       rhs_dims, out_dims, queue, events);
}

template <typename Op, typename T, template <typename> class MemObj>
SNNStatus launch_binaryop(MemObj<T const>& lhs, MemObj<T const>& rhs,
                          MemObj<T>& out, const std::vector<int>& lhs_dims,
                          const std::vector<int>& rhs_dims,
                          cl::sycl::queue& queue,
                          const std::vector<cl::sycl::event>& events) {
  std::vector<int> out_dims;
  auto status = compute_out_dims(lhs_dims, rhs_dims, out_dims);
  if (status.status != StatusCode::OK) return status;
  return launch_binaryop<Op>(lhs, rhs, out, lhs_dims, rhs_dims, out_dims, queue,
                             events);
}

template <typename Op, typename T, template <typename> class MemObj>
SNNStatus launch_binaryop(MemObj<T const>& lhs, MemObj<T const>& rhs,
                          MemObj<T>& out, const std::vector<int>& dims,
                          cl::sycl::queue& queue,
                          const std::vector<cl::sycl::event>& events) {
  return launch_binaryop<Op>(lhs, rhs, out, dims, dims, dims, queue, events);
}

template <typename Op, typename T, template <typename> class MemObj>
SNNStatus launch_binaryop(MemObj<T const>& lhs, MemObj<T const>& rhs,
                          MemObj<T>& out, int size, cl::sycl::queue& queue,
                          const std::vector<cl::sycl::event>& events) {
  return launch_binaryop<Op>(lhs, rhs, out, std::vector<int>{size}, queue,
                             events);
}

}  // namespace internal
}  // namespace binaryop
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_BINARYOP_LAUNCH_INTERNAL_H_

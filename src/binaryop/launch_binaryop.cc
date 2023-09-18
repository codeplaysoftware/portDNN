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

#include <CL/sycl.hpp>
#include <utility>

#include "portdnn/binaryop/operators.h"
#include "portdnn/export.h"
#include "portdnn/internal/binaryop/launch.h"
#include "src/binaryop/kernels.h"
#include "src/binaryop/queue_binaryop_kernel.h"

namespace sycldnn {
namespace binaryop {

namespace internal {

template <typename T, typename Op, int VectorWidth,
          template <typename> class MemObj,
          typename = std::enable_if<is_mem_obj_v<MemObj<T>, T>>>
SNNStatus launch_vec_kernel_with_vec_width(
    MemObj<T const>& lhs, MemObj<T const>& rhs, MemObj<T>& out, bool bcast_lhs,
    const std::vector<int>& lhs_dims, const std::vector<int>& rhs_dims,
    const std::vector<int>& out_dims, cl::sycl::queue& queue,
    const std::vector<cl::sycl::event>& events) {
  constexpr bool is_usm = is_usm_obj_v<MemObj<T>, T>;
  if (lhs_dims.size() == 1) {
    return queue_binaryop<BinaryOpVec<T, Op, int, VectorWidth, is_usm>>(
        lhs, rhs, out, lhs_dims, rhs_dims, out_dims, queue, events);
  } else if (lhs_dims.size() == 2) {
    if (bcast_lhs) {
      return queue_binaryop<
          BinaryOpBcastLhsVec2D<T, Op, int, VectorWidth, is_usm>>(
          lhs, rhs, out, lhs_dims, rhs_dims, out_dims, queue, events);
    } else {
      return queue_binaryop<
          BinaryOpBcastRhsVec2D<T, Op, int, VectorWidth, is_usm>>(
          lhs, rhs, out, lhs_dims, rhs_dims, out_dims, queue, events);
    }
  } else {
    SNN_ASSERT(lhs_dims.size() == 3,
               "Invalid internal dimensions for BinaryOp operands");
    if (bcast_lhs) {
      return queue_binaryop<
          BinaryOpBcastLhsVec3D<T, Op, int, VectorWidth, is_usm>>(
          lhs, rhs, out, lhs_dims, rhs_dims, out_dims, queue, events);
    } else {
      return queue_binaryop<
          BinaryOpBcastRhsVec3D<T, Op, int, VectorWidth, is_usm>>(
          lhs, rhs, out, lhs_dims, rhs_dims, out_dims, queue, events);
    }
  }
}

template <typename T, typename Op, template <typename> class MemObj,
          typename = std::enable_if<is_mem_obj_v<MemObj<T>, T>>>
SNNStatus launch_vec_kernel(MemObj<T const>& lhs, MemObj<T const>& rhs,
                            MemObj<T>& out, bool bcast_lhs,
                            const std::vector<int>& lhs_dims,
                            const std::vector<int>& rhs_dims,
                            const std::vector<int>& out_dims,
                            cl::sycl::queue& queue,
                            const std::vector<cl::sycl::event>& events) {
  if (out_dims.back() % 4 == 0) {
    return launch_vec_kernel_with_vec_width<T, Op, 4>(
        lhs, rhs, out, bcast_lhs, lhs_dims, rhs_dims, out_dims, queue, events);
  } else if (out_dims.back() % 2 == 0) {
    return launch_vec_kernel_with_vec_width<T, Op, 2>(
        lhs, rhs, out, bcast_lhs, lhs_dims, rhs_dims, out_dims, queue, events);
  } else {
    return launch_vec_kernel_with_vec_width<T, Op, 1>(
        lhs, rhs, out, bcast_lhs, lhs_dims, rhs_dims, out_dims, queue, events);
  }
}

template <typename Op, typename T, template <typename> class MemObj>
SNNStatus launch_binaryop(MemObj<T const>& lhs, MemObj<T const>& rhs,
                          MemObj<T>& out, std::vector<int> lhs_dims,
                          std::vector<int> rhs_dims,
                          const std::vector<int>& out_dims,
                          cl::sycl::queue& queue,
                          const std::vector<cl::sycl::event>& events) {
  SNN_VALIDATE_PARAM(lhs.get_extent() == helpers::get_total_size(lhs_dims),
                     "Mismatching number of lhs elements");
  SNN_VALIDATE_PARAM(rhs.get_extent() == helpers::get_total_size(rhs_dims),
                     "Mismatching number of rhs elements");
  SNN_VALIDATE_PARAM(out.get_extent() == helpers::get_total_size(out_dims),
                     "Mismatching number of out elements");
  constexpr bool is_usm = is_usm_obj_v<MemObj<T>, T>;
  size_t num_dims = out_dims.size();
  while (lhs_dims.size() < num_dims) {
    lhs_dims.insert(lhs_dims.begin(), 1);
  }
  while (rhs_dims.size() < num_dims) {
    rhs_dims.insert(rhs_dims.begin(), 1);
  }

  // Fold continuous non-broadcasted dimensions and broadcasted dimensions.
  // This simplifies kernels indices computations.
  // Broadcast direction is used to differentiate consecutive dimensions.
  // Consecutive dimensions in the same "direction" can be folded.
  std::vector<int> folded_lhs_dims{lhs_dims[0]};
  std::vector<int> folded_rhs_dims{rhs_dims[0]};
  std::vector<int> folded_out_dims{out_dims[0]};
  auto get_bcast_dir = [&](int i) {
    return lhs_dims[i] == rhs_dims[i] ? 0 : (lhs_dims[i] == 1 ? -1 : 1);
  };
  int prev_bcast_dir = get_bcast_dir(0);
  for (size_t i = 1; i < num_dims; ++i) {
    int bcast_dir = get_bcast_dir(i);
    if (prev_bcast_dir == bcast_dir) {
      folded_lhs_dims.back() *= lhs_dims[i];
      folded_rhs_dims.back() *= rhs_dims[i];
      folded_out_dims.back() *= out_dims[i];
    } else {
      folded_lhs_dims.push_back(lhs_dims[i]);
      folded_rhs_dims.push_back(rhs_dims[i]);
      folded_out_dims.push_back(out_dims[i]);
    }
    prev_bcast_dir = bcast_dir;
  }

  // Store the broadcasted index and whether the lhs operand is the one
  // broadcasted. This is used to choose which implementation to choose.
  std::vector<std::pair<int, bool>> broadcasted_dims;
  for (size_t i = 0; i < folded_out_dims.size(); ++i) {
    if (folded_lhs_dims[i] != folded_rhs_dims[i]) {
      SNN_ASSERT(folded_lhs_dims[i] == 1 || folded_rhs_dims[i] == 1,
                 "Invalid internal dimensions for BinaryOp operands");
      broadcasted_dims.emplace_back(i, folded_lhs_dims[i] == 1);
    }
  }

  if (broadcasted_dims.size() == 0) {
    SNN_ASSERT(folded_out_dims.size() == 1,
               "Failed to fold BinaryOp dimensions");
    return launch_vec_kernel<T, Op>(lhs, rhs, out, false, folded_lhs_dims,
                                    folded_rhs_dims, folded_out_dims, queue,
                                    events);
  } else if (broadcasted_dims.size() == 1) {
    // Vectorize on the last dimension of the operands.
    // Set the number of dimensions to 2 or 3 to simplify the kernels.
    if (broadcasted_dims[0].first == 0) {
      folded_lhs_dims.insert(folded_lhs_dims.begin(), 1);
      folded_rhs_dims.insert(folded_rhs_dims.begin(), 1);
      folded_out_dims.insert(folded_out_dims.begin(), 1);
      broadcasted_dims[0].first = 1;
    }
    // Remove the last dimension if it is size 1 for better vectorization
    if (folded_out_dims.size() == 3 && folded_out_dims.back() == 1) {
      folded_lhs_dims.pop_back();
      folded_rhs_dims.pop_back();
      folded_out_dims.pop_back();
    }
    SNN_ASSERT(folded_lhs_dims.size() == folded_out_dims.size(),
               "Invalid internal dimensions for BinaryOp operands");
    SNN_ASSERT(folded_rhs_dims.size() == folded_out_dims.size(),
               "Invalid internal dimensions for BinaryOp operands");
    SNN_ASSERT(folded_out_dims.size() == 2 || folded_out_dims.size() == 3,
               "Invalid internal dimensions for BinaryOp operands");
    return launch_vec_kernel<T, Op>(lhs, rhs, out, broadcasted_dims[0].second,
                                    folded_lhs_dims, folded_rhs_dims,
                                    folded_out_dims, queue, events);
  }

  // Fallback to generic implementation
  return queue_binaryop<BinaryOp<T, Op, int, is_usm>>(
      lhs, rhs, out, folded_lhs_dims, folded_rhs_dims, folded_out_dims, queue,
      events);
}

#define INSTANTIATE_BINARYOP_LAUNCH(DTYPE, OP, MEMOBJ)                      \
  template SNN_EXPORT SNNStatus launch_binaryop<OP, DTYPE>(                 \
      MEMOBJ<DTYPE const> & inp1_access, MEMOBJ<DTYPE const> & inp2_access, \
      MEMOBJ<DTYPE> & outp_access, std::vector<int> lhs_dims,               \
      std::vector<int> rhs_dims, const std::vector<int>& out_dims,          \
      cl::sycl::queue& queue, const std::vector<cl::sycl::event>& events);

#define INSTANTIATE_BINARYOP_FOR_TYPE(DTYPE, MEMOBJ) \
  INSTANTIATE_BINARYOP_LAUNCH(DTYPE, Add, MEMOBJ)    \
  INSTANTIATE_BINARYOP_LAUNCH(DTYPE, Sub, MEMOBJ)    \
  INSTANTIATE_BINARYOP_LAUNCH(DTYPE, Mul, MEMOBJ)    \
  INSTANTIATE_BINARYOP_LAUNCH(DTYPE, Div, MEMOBJ)

#ifdef SNN_ENABLE_USM
INSTANTIATE_BINARYOP_FOR_TYPE(float, USMMemObject);
#endif
INSTANTIATE_BINARYOP_FOR_TYPE(float, BufferMemObject);

#ifdef SNN_USE_HALF
#ifdef SNN_ENABLE_USM
INSTANTIATE_BINARYOP_FOR_TYPE(cl::sycl::half, USMMemObject);
#endif
INSTANTIATE_BINARYOP_FOR_TYPE(cl::sycl::half, BufferMemObject);
#endif  // SNN_USE_HALF

#ifdef SNN_USE_DOUBLE
#ifdef SNN_ENABLE_USM
INSTANTIATE_BINARYOP_FOR_TYPE(double, USMMemObject);
#endif
INSTANTIATE_BINARYOP_FOR_TYPE(double, BufferMemObject);
#endif  // SNN_USE_DOUBLE

}  // namespace internal
}  // namespace binaryop
}  // namespace sycldnn

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
#ifndef SYCLDNN_INCLUDE_REDUCE_LAUNCH_H_
#define SYCLDNN_INCLUDE_REDUCE_LAUNCH_H_

/**
 * \file
 * Implements the \ref sycldnn::reduce::launch() function, which asynchronously
 * dispatches the SYCL kernels required to perform reductions.
 */
#include <type_traits>

#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include "sycldnn/internal/helpers/types.h"
#include "sycldnn/internal/reduce/launch.h"
#include "sycldnn/reduce/operators.h"

namespace sycldnn {
namespace reduce {
/**
 * Launch a reduction of [batch, outer, inner] applying Op on the outer
 * dimension. The output shape is [batch, inner].
 *
 * \tparam Op Operation to apply on the reduced dimension
 * \param input A pointer to the memory representing the input tensor.
 * \param output A pointer to the memory representing the output tensor.
 * \param batches The number of batches. Must be a positive value.
 * \param outer Outer size. This is the dimension that is always reduced. Must
 * be a positive value.
 * \param inner Inner size. Must be a positive value.
 * \param backend The backend implementation, used to map between pointer
 *                representations.
 * \return Returns an SNNStatus containing the SYCL event tied to the kernel
 *         launches and a StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */
template <typename T, typename Op, typename Backend>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T> output, int batches,
                 int outer, int inner, Backend& backend) {
  static_assert(std::is_same<Op, reduce::Add>::value ||
                    std::is_same<Op, reduce::Mean>::value,
                "Invalid Reduction Type");
  SNN_VALIDATE_PARAM(batches > 0, "The number of batches must be positive.");
  SNN_VALIDATE_PARAM(outer > 0, "The value of outer must be positive.");
  SNN_VALIDATE_PARAM(inner > 0, "The value of inner must be positive.");

  size_t in_size = batches * outer * inner;
  size_t out_size = batches * inner;

  auto in_acc = backend.get_mem_object(input, in_size);
  auto out_acc = backend.get_mem_object(output, out_size);

  auto sycl_queue = backend.get_queue();
  auto program = backend.get_program();
  bool supports_subgroup = backend.supports_subgroup();
  auto& max_kernel_sub_group_sizes = backend.get_max_kernel_sub_group_sizes();

  return internal::launch<T, Op>(in_acc, out_acc, batches, outer, inner,
                                 sycl_queue, program, supports_subgroup,
                                 max_kernel_sub_group_sizes);
}
}  // namespace reduce
}  // namespace sycldnn
#endif  // SYCLDNN_INCLUDE_REDUCE_LAUNCH_H_

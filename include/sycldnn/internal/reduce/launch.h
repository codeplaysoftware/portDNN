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
#include <string>
#include <unordered_map>

#include "sycldnn/export.h"
#include "sycldnn/internal/helpers/types.h"
#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

namespace sycldnn {
namespace reduce {
namespace internal {

/**
 * The internal reduce launcher.
 *
 * Implemented in the compiled SYCL DNN library.
 */
#ifdef SNN_DISABLE_SYCL_PROGRAM
template <typename T, typename Op>
SNN_EXPORT SNNStatus launch(BaseMemObject<T const>& input,
                            BaseMemObject<T>& output, int batches, int outer,
                            int inner, cl::sycl::queue& queue);
#else
template <typename T, typename Op>
SNN_EXPORT SNNStatus launch(BaseMemObject<T const>& input,
                            BaseMemObject<T>& output, int batches, int outer,
                            int inner, cl::sycl::queue& queue,
                            cl::sycl::program& program, bool supports_subgroup,
                            sycldnn::internal::types::KernelSubgroupSizesMap&
                                max_kernel_sub_group_sizes);
#endif
/**
 * Helper for internal reduce launcher.
 */
#ifdef SNN_DISABLE_SYCL_PROGRAM
template <typename Op, typename T, typename Backend>
inline SNNStatus launch(BaseMemObject<T const>& input, BaseMemObject<T>& output,
                        int batches, int outer, int inner, Backend& backend) {
  auto queue = backend.get_queue();
  return launch<T, Op>(input, output, batches, outer, inner, queue);
}

#else
template <typename Op, typename T, typename Backend>
inline SNNStatus launch(BaseMemObject<T const>& input, BaseMemObject<T>& output,
                        int batches, int outer, int inner, Backend& backend) {
  auto queue = backend.get_queue();
  auto program = backend.get_program();
  bool supports_subgroup = backend.supports_subgroup();
  auto& max_kernel_sub_group_sizes = backend.get_max_kernel_sub_group_sizes();
  return launch<T, Op>(input, output, batches, outer, inner, queue, program,
                       supports_subgroup, max_kernel_sub_group_sizes);
}
#endif

}  // namespace internal
}  // namespace reduce
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_INTERNAL_REDUCE_LAUNCH_H_

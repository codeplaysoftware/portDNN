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
#ifndef PORTDNN_INCLUDE_INTERNAL_REDUCE_LAUNCH_H_
#define PORTDNN_INCLUDE_INTERNAL_REDUCE_LAUNCH_H_

#include <CL/sycl.hpp>
#include <string>
#include <unordered_map>

#include "portdnn/export.h"
#include "portdnn/internal/helpers/types.h"
#include "portdnn/mem_object.h"
#include "portdnn/reduce/operators.h"
#include "portdnn/status.h"

namespace sycldnn {
namespace reduce {
namespace internal {

/**
 * The internal reduce launcher.
 *
 * Implemented in the compiled SYCL DNN library.
 */
#ifdef SNN_DISABLE_SYCL_PROGRAM
template <typename T, typename Op, template <typename> class mem_obj>
SNN_EXPORT SNNStatus launch(mem_obj<T const>& input, mem_obj<T>& output,
                            int batches, int outer, int inner,
                            cl::sycl::queue& queue,
                            const std::vector<cl::sycl::event>& events);
#else
template <typename T, typename Op, template <typename> class mem_obj>
SNN_EXPORT SNNStatus launch(mem_obj<T const>& input, mem_obj<T>& output,
                            int batches, int outer, int inner,
                            cl::sycl::queue& queue, cl::sycl::program& program,
                            bool supports_subgroup,
                            sycldnn::internal::types::KernelSubgroupSizesMap&
                                max_kernel_sub_group_sizes,
                            const std::vector<cl::sycl::event>& events);
#endif

/**
 * Forward declarations
 */
template <typename Op, typename T, typename Backend,
          template <typename> class mem_obj>
inline SNNStatus launch(mem_obj<T const>& input, mem_obj<T>& output,
                        int batches, int outer, int inner, Backend& backend,
                        const std::vector<cl::sycl::event>& events);

/**
 * The internal tensor transpose sublauncher.
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

  return internal::launch<Op>(in_acc, out_acc, batches, outer, inner, backend,
                              events);
}

/**
 * Helper for internal reduce launcher.
 */
#ifdef SNN_DISABLE_SYCL_PROGRAM
template <typename Op, typename T, typename Backend,
          template <typename> class mem_obj>
inline SNNStatus launch(mem_obj<T const>& input, mem_obj<T>& output,
                        int batches, int outer, int inner, Backend& backend,
                        const std::vector<cl::sycl::event>& events) {
  auto queue = backend.get_queue();
  return launch<T, Op>(input, output, batches, outer, inner, queue, events);
}

#else
template <typename Op, typename T, typename Backend,
          template <typename> class mem_obj>
inline SNNStatus launch(mem_obj<T const>& input, mem_obj<T>& output,
                        int batches, int outer, int inner, Backend& backend,
                        const std::vector<cl::sycl::event>& events) {
  auto queue = backend.get_queue();
  auto program = backend.get_program();
  bool supports_subgroup = backend.supports_subgroup();
  auto& max_kernel_sub_group_sizes = backend.get_max_kernel_sub_group_sizes();
  return launch<T, Op>(input, output, batches, outer, inner, queue, program,
                       supports_subgroup, max_kernel_sub_group_sizes, events);
}
#endif

}  // namespace internal
}  // namespace reduce
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_INTERNAL_REDUCE_LAUNCH_H_

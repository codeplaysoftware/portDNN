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
#ifndef PORTDNN_SRC_REDUCE_QUEUE_REDUCTION_IMPL_H_
#define PORTDNN_SRC_REDUCE_QUEUE_REDUCTION_IMPL_H_

#include <limits>
#include <type_traits>

#include "portdnn/mem_object.h"
#include "portdnn/status.h"

#include "src/helpers/math.h"
#include "src/reduce/default_kernel.h"
#include "src/reduce/queue_reduction.h"

#include "portdnn/helpers/mem_utils.h"

#ifndef SNN_DISABLE_SYCL_PROGRAM
#include "src/reduce/subgroup_kernel.h"
#endif

namespace sycldnn {
namespace reduce {
namespace internal {

template <class T, class Op>
static constexpr T init_val = 0;

template <class T>
static constexpr T init_val<T, Max> = std::numeric_limits<T>::min();

template <class T>
static constexpr T init_val<T, Min> = std::numeric_limits<T>::max();

template <typename T, typename Index, typename Op,
          template <typename> class MemObj>
SNNStatus queue_default_kernel(MemObj<T const>& input_mem,
                               MemObj<T>& output_mem, int batches, int outer,
                               int inner, int finalizeParam,
                               cl::sycl::queue& queue,
                               const std::vector<cl::sycl::event>& events) {
  constexpr bool is_usm = is_usm_obj_v<MemObj<T>, T>;
  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    cgh.depends_on(events);
    auto input = input_mem.read_mem(cgh);
    auto output = output_mem.write_mem(cgh);

    ReduceKernel<T, Index, Op, is_usm> functor{
        input, output, batches, outer, inner, finalizeParam, init_val<T, Op>};

    cgh.parallel_for(cl::sycl::range<2>(batches, inner), functor);
  });
  return {event, StatusCode::OK};
}

#ifndef SNN_DISABLE_SYCL_PROGRAM
template <typename T, typename Index, typename Op,
          template <typename> class MemObj>
SNNStatus queue_subgroup_kernel(
    MemObj<T const>& input_mem, MemObj<T>& output_mem, int batches, int outer,
    int inner, cl::sycl::queue& queue, cl::sycl::program& program,
    sycldnn::internal::types::KernelSubgroupSizesMap&
        max_kernel_sub_group_sizes,
    const std::vector<cl::sycl::event>& events) {
  constexpr bool is_usm = is_usm_obj_v<MemObj<T>, T>;
  using Kernel = ReduceSubgroupKernel<T, Index, Op, is_usm>;
  using namespace sycldnn::helpers::math;
  auto device = queue.get_device();
  const size_t max_work_group_size =
      device.get_info<cl::sycl::info::device::max_work_group_size>();
  const auto max_work_item_sizes =
#ifndef SYCL_IMPLEMENTATION_ONEAPI
      device.get_info<cl::sycl::info::device::max_work_item_sizes>();
#else
      device.get_info<cl::sycl::info::device::max_work_item_sizes<3> >();
#endif
  size_t alignment = std::min(max_work_item_sizes[0], max_work_group_size);

  auto fallback = [&](MemObj<T const>& input, size_t outer_size) {
    return queue_default_kernel<T, Index, Op>(
        input, output_mem, batches, outer_size, inner, outer, queue, events);
  };
  auto query_subgroup_size = [&](cl::sycl::kernel kernel,
                                 const cl::sycl::range<2>& local_range) {
    return kernel.template get_sub_group_info<
        cl::sycl::info::kernel_sub_group::max_sub_group_size_for_ndrange>(
        device, cl::sycl::range<3>(1, local_range[0], local_range[1]));
  };

  size_t max_sub_group_size;
  static const std::string kernelName = typeid(Kernel).name();
  if (max_kernel_sub_group_sizes.find(kernelName) !=
      max_kernel_sub_group_sizes.end()) {
    max_sub_group_size = max_kernel_sub_group_sizes[kernelName];
  } else {
    program.build_with_kernel_type<Kernel>();
    max_sub_group_size = query_subgroup_size(program.get_kernel<Kernel>(),
                                             cl::sycl::range<2>(1, alignment));
    max_kernel_sub_group_sizes.insert({kernelName, max_sub_group_size});
  }
  cl::sycl::kernel kernel = program.get_kernel<Kernel>();

  if (max_sub_group_size == 1) return fallback(input_mem, outer);

  cl::sycl::range<2> input_range(batches, outer);
  cl::sycl::range<2> kernel_range = input_range;
  cl::sycl::range<2> local_wg_range(1, 1);
  auto update_local_range = [&]() {
    if (kernel_range[1] < alignment) {
      local_wg_range[1] = kernel_range[1];
    } else {
      kernel_range[1] = align(kernel_range[1], max_sub_group_size);
      local_wg_range[1] = max_sub_group_size;
      size_t multiple = kernel_range[1] / max_sub_group_size;
      for (int i = multiple; i > 1; --i) {
        size_t new_size = local_wg_range[1] * i;
        if (new_size <= alignment && kernel_range[1] % new_size == 0) {
          local_wg_range[1] = new_size;
          break;
        }
      }
    }
  };
  update_local_range();

  size_t sub_group_size = query_subgroup_size(kernel, local_wg_range);
  if (sub_group_size <= 1) return fallback(input_mem, outer);

  size_t reduce_size = input_range[1];
  size_t next_reduce_size = divide_ceil(input_range[1], sub_group_size);

  // TODO: Implement this using workspace
  cl::sycl::range<2> mem1_size(input_range[0], next_reduce_size);
  cl::sycl::range<2> mem2_size(input_range[0],
                               divide_ceil(next_reduce_size, sub_group_size));
  auto sycl_MemObj = sycldnn::helpers::alloc<T, is_usm>(
      mem1_size.size() + mem2_size.size(), queue);

  auto mem1 = make_mem_object(sycl_MemObj, mem1_size.size(), 0);
  auto mem2 = make_mem_object(sycl_MemObj, mem2_size.size(), mem1_size.size());

  cl::sycl::nd_range<2> nd_range0(kernel_range, local_wg_range);
  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    auto in_mem = input_mem.read_mem(cgh);
    auto out_mem =
        next_reduce_size == 1 ? output_mem.write_mem(cgh) : mem1.write_mem(cgh);
    size_t out_size1 = out_mem.get_extent() / input_range[0];
    Kernel functor(in_mem, out_mem, sub_group_size, reduce_size, input_range[1],
                   out_size1);
    cgh.parallel_for(kernel, nd_range0, functor);
  });
  int iter = 0;
  while (next_reduce_size > 1) {
    reduce_size = next_reduce_size;
    kernel_range[1] = divide_ceil(kernel_range[1], sub_group_size);
    update_local_range();
    sub_group_size = query_subgroup_size(kernel, local_wg_range);
    next_reduce_size = divide_ceil(next_reduce_size, sub_group_size);
    auto mem_in = iter % 2 == 0 ? mem1.as_const() : mem2.as_const();
    // Finish the reduction with the default kernel if the local_wg_range is not
    // suitable to subgroups anymore.
    if (sub_group_size <= 1) {
      SNNStatus status = fallback(mem_in, reduce_size);
      sycldnn::helpers::enqueue_free(queue, {status.event}, sycl_MemObj);
      return status;
    }
    cl::sycl::nd_range<2> nd_range_iter(kernel_range, local_wg_range);
    event = queue.submit([&](cl::sycl::handler& cgh) {
      auto& mem_out = iter % 2 == 0 ? mem2 : mem1;
      auto in_mem = mem_in.read_mem(cgh);
      auto out_mem =
          (next_reduce_size == 1 ? output_mem : mem_out).write_mem(cgh);
      size_t in_size1 = in_mem.get_extent() / input_range[0];
      size_t out_size1 = out_mem.get_extent() / input_range[0];
      Kernel functor(in_mem, out_mem, sub_group_size, reduce_size, in_size1,
                     out_size1);
      cgh.parallel_for(kernel, nd_range_iter, functor);
    });
    ++iter;
  }
  if (SubgroupReducer<T, Index, Op>::RequireFinalize) {
    event = queue.submit([&](cl::sycl::handler& cgh) {
      auto out_mem = output_mem.read_write_mem(cgh);
      ReduceFinalize<T, Index, Op, is_usm> functor(out_mem, outer);
      cgh.parallel_for(cl::sycl::range<1>(out_mem.get_extent()), functor);
    });
  }
  sycldnn::helpers::enqueue_free(queue, {event}, sycl_MemObj);
  return {event, StatusCode::OK};
}
#endif

}  // namespace internal
}  // namespace reduce
}  // namespace sycldnn
#endif  // PORTDNN_SRC_REDUCE_QUEUE_REDUCTION_IMPL_H_

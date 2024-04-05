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
#ifndef SYCLDNN_SRC_REDUCE_LOCAL_MEM_KERNEL_H_
#define SYCLDNN_SRC_REDUCE_LOCAL_MEM_KERNEL_H_

#include "sycldnn/accessor_types.h"
#include "sycldnn/status.h"

#include "sycldnn/reduce/operators.h"

#include "sycldnn/internal/reduce/identities.h"

#include "src/helpers/local_mem_tile.h"

namespace sycldnn {
namespace reduce {

template <typename T, typename Op>
struct ReduceOp;

template <typename T>
struct ReduceOp<T, Mean> {
  static auto constexpr Identity = internal::Identity<T, Mean>::value;
  static auto constexpr Op = cl::sycl::plus<T>();
  static bool constexpr NeedsFinalize = true;

  template <cl::sycl::memory_order Order, cl::sycl::memory_scope Scope,
            cl::sycl::access::address_space Space, typename Index>
  static void atomic_op(T& ref, T val, Index size) {
    auto atomic_out = cl::sycl::atomic_ref<T, Order, Scope, Space>(ref);
    atomic_out.fetch_add(val / size);
  }

  template <typename Index>
  static void finalize(T& ref, Index size) {
    ref /= size;
  }
};

template <typename T>
struct ReduceOp<T, Add> {
  static auto constexpr Identity = internal::Identity<T, Add>::value;
  static auto constexpr Op = cl::sycl::plus<T>();
  static bool constexpr NeedsFinalize = false;

  template <cl::sycl::memory_order Order, cl::sycl::memory_scope Scope,
            cl::sycl::access::address_space Space, typename Index>
  static void atomic_op(T& ref, T val, Index) {
    auto atomic_out = cl::sycl::atomic_ref<T, Order, Scope, Space>(ref);
    atomic_out.fetch_add(val);
  }

  template <typename Index>
  static void finalize(T&, Index) {}
};

template <typename T>
struct ReduceOp<T, Max> {
  static auto constexpr Identity = internal::Identity<T, Max>::value;
  static auto constexpr Op = cl::sycl::maximum<T>();
  static bool constexpr NeedsFinalize = false;

  template <cl::sycl::memory_order Order, cl::sycl::memory_scope Scope,
            cl::sycl::access::address_space Space, typename Index>
  static void atomic_op(T& ref, T val, Index) {
    auto atomic_out = cl::sycl::atomic_ref<T, Order, Scope, Space>(ref);
    atomic_out.fetch_max(val);
  }

  template <typename Index>
  static void finalize(T&, Index) {}
};

template <typename T>
struct ReduceOp<T, Min> {
  static auto constexpr Identity = internal::Identity<T, Min>::value;
  static auto constexpr Op = cl::sycl::minimum<T>();
  static bool constexpr NeedsFinalize = false;

  template <cl::sycl::memory_order Order, cl::sycl::memory_scope Scope,
            cl::sycl::access::address_space Space, typename Index>
  static void atomic_op(T& ref, T val, Index) {
    auto atomic_out = cl::sycl::atomic_ref<T, Order, Scope, Space>(ref);
    atomic_out.fetch_min(val);
  }

  template <typename Index>
  static void finalize(T&, Index) {}
};

template <typename T, typename Index, typename Op, bool Transpose,
          int NumSGReductions, bool GlobalAtomicReduce, bool BankOffset,
          bool IsUSM, typename LocalMem>
struct LocalMemReduceKernel {
  using ReduceOp = ReduceOp<T, Op>;
  using MemTile =
      helpers::internal::LocalMemTile<T, Index, 3, true, BankOffset>;
  // This is used to solve the numerical issue of Mean atomics
  // static bool constexpr WIReduce = false;
  static bool constexpr WIReduce =
      std::is_same_v<Op, reduce::Mean> && NumSGReductions == 0;
  static auto constexpr Order = cl::sycl::memory_order::relaxed;
  static auto constexpr DeviceScope = cl::sycl::memory_scope::device;
  static auto constexpr WGScope = cl::sycl::memory_scope::work_group;
  static auto constexpr GlobalSpace =
      cl::sycl::access::address_space::global_space;
  static auto constexpr LocalSpace =
      cl::sycl::access::address_space::local_space;

  LocalMemReduceKernel(ReadMem<T const, IsUSM> const& input,
                       ReadWriteMem<T, IsUSM> const& output,
                       LocalMem& local_mem, Index batch_size, Index reduce_size,
                       Index stride_size, Index row_tile, Index col_tile,
                       Index batch_groups)
      : input_{input},
        output_{output},
        local_mem_(local_mem),
        batch_size_{batch_size},
        reduce_size_{reduce_size},
        stride_size_{stride_size},
        row_tile_{row_tile},
        col_tile_{col_tile},
        batch_groups_{batch_groups} {}

  void SNN_ALWAYS_INLINE operator()(cl::sycl::nd_item<3> nd_item) const {
    auto const in_ptr = input_.get_pointer();
    auto out_ptr = output_.get_pointer();
    auto local_ptr = local_mem_.get_pointer();

    MemTile tile(row_tile_, col_tile_, reduce_size_, stride_size_, nd_item,
                 ReduceOp::Identity);

    // Copy values from global memory to local
    tile.load_memory(in_ptr, local_ptr);

    if constexpr (Transpose) {
      // Transpose data in local memory so the reduction is coalesced
      tile.transpose(local_ptr);
    }

    // Reduce values using subgroup reducer
    tile.template sg_reduce<ReduceOp, Transpose, NumSGReductions>(local_ptr);

    if constexpr (WIReduce) {
      // Reduce values using work-item reducer
      tile.template wi_reduce<ReduceOp, Transpose>(local_ptr);
    }

    if constexpr (GlobalAtomicReduce) {
      // Reduce remaining values into global memory using global atomics
      tile.template atomic_reduce<ReduceOp, Transpose, Order, DeviceScope,
                                  GlobalSpace>(
          local_ptr, out_ptr, reduce_size_ * batch_size_ / batch_groups_,
          batch_groups_);
    } else {
      // Finalize reduction
      tile.template finalize_reduce<ReduceOp, Transpose>(local_ptr,
                                                         reduce_size_);
      // Store results from local memory to global
      tile.template store_memory<true, Transpose>(local_ptr, out_ptr);
    }
  }

 private:
  ReadMem<T const, IsUSM> input_;
  ReadWriteMem<T, IsUSM> output_;
  LocalMem local_mem_;
  Index const batch_size_;
  Index const reduce_size_;
  Index const stride_size_;
  Index const row_tile_;
  Index const col_tile_;
  Index const batch_groups_;
};
}  // namespace reduce
}  // namespace sycldnn
#endif

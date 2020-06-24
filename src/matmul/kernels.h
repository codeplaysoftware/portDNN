/*
 * Copyright 2018 Codeplay Software Ltd
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
#ifndef SYCLDNN_SRC_MATMUL_KERNELS_H_
#define SYCLDNN_SRC_MATMUL_KERNELS_H_

#include "sycldnn/accessor_types.h"
#include "sycldnn/status.h"

#include "src/matmul/blocks.h"

namespace sycldnn {
namespace matmul {
template <typename T, typename Index, bool TransposeLHS, bool TransposeRHS,
          int RowTile, int AccTile, int ColTile>
struct MatmulKernel {
  MatmulKernel(ReadAccessor<T const> const& lhs,
               ReadAccessor<T const> const& rhs,
               ReadWriteAccessor<T> const& output, Index batches, Index m,
               Index k, Index n, T beta)
      : lhs_{lhs},
        rhs_{rhs},
        output_{output},
        batches_{batches},
        m_{m},
        k_{k},
        n_{n},
        beta_{beta} {}

  void SNN_ALWAYS_INLINE operator()(cl::sycl::nd_item<3> item) {
    Index batch = item.get_global_id(0);
    Index row = item.get_global_id(1) * RowTile;
    Index col = item.get_global_id(2) * ColTile;

    if (row < m_ && col < n_) {
      auto lhs_ptr = lhs_.get_pointer() + batch * m_ * k_;
      auto rhs_ptr = rhs_.get_pointer() + batch * k_ * n_;
      auto out_ptr = output_.get_pointer() + batch * m_ * n_;

      auto out_block = VectorBlock<T, RowTile, ColTile>{};
      if (beta_ != static_cast<T>(0)) {
        // Convert out_ptr from multi_ptr<T> to multi_ptr<T const>
        auto const_out_ptr =
            cl::sycl::multi_ptr<T const,
                                cl::sycl::access::address_space::global_space>{
                out_ptr.get()};
        out_block =
            load<RowTile, ColTile, false>(const_out_ptr, row, col, m_, n_);
        scalar_multiply(out_block, beta_);
      }

      for (Index acc_idx = 0; acc_idx < k_; acc_idx += AccTile) {
        auto lhs_block =
            load<RowTile, AccTile, TransposeLHS>(lhs_ptr, row, acc_idx, m_, k_);
        auto rhs_block =
            load<AccTile, ColTile, TransposeRHS>(rhs_ptr, acc_idx, col, k_, n_);
        block_mmacc(lhs_block, rhs_block, out_block);
      }
      store_block(out_block, out_ptr, row, col, n_, m_, n_);
    }
  }

 private:
  ReadAccessor<T const> lhs_;
  ReadAccessor<T const> rhs_;
  ReadWriteAccessor<T> output_;
  Index const batches_;
  Index const m_;
  Index const k_;
  Index const n_;
  T const beta_;
};

template <typename T, typename Index, bool TransposeLHS, bool TransposeRHS,
          int RowTile, int AccTile, int ColTile>
struct MatmulUncheckedKernel {
 public:
  MatmulUncheckedKernel(ReadAccessor<T const> const& lhs,
                        ReadAccessor<T const> const& rhs,
                        ReadWriteAccessor<T> const& output, Index batches,
                        Index m, Index k, Index n, T beta)
      : lhs_{lhs},
        rhs_{rhs},
        output_{output},
        batches_{batches},
        m_{m},
        k_{k},
        n_{n},
        beta_{beta} {}

  void SNN_ALWAYS_INLINE operator()(cl::sycl::nd_item<3> item) {
    Index batch = item.get_global_id(0);
    Index row = item.get_global_id(1) * RowTile;
    Index col = item.get_global_id(2) * ColTile;

    if (row < m_ && col < n_) {
      auto lhs_ptr = lhs_.get_pointer() + batch * m_ * k_;
      auto rhs_ptr = rhs_.get_pointer() + batch * k_ * n_;
      auto out_ptr = output_.get_pointer() + batch * m_ * n_;

      auto const lhs_ld = TransposeLHS ? m_ : k_;
      auto const rhs_ld = TransposeRHS ? k_ : n_;
      auto const out_ld = n_;

      auto out_block = VectorBlock<T, RowTile, ColTile>{};
      if (beta_ != static_cast<T>(0)) {
        // Convert out_ptr from multi_ptr<T> to multi_ptr<T const>
        auto const_out_ptr =
            cl::sycl::multi_ptr<T const,
                                cl::sycl::access::address_space::global_space>{
                out_ptr.get()};
        static constexpr bool Transpose = false;
        out_block =
            load<RowTile, ColTile, Transpose>(const_out_ptr, row, col, out_ld);
        scalar_multiply(out_block, beta_);
      }

      for (Index acc_idx = 0; acc_idx < k_; acc_idx += AccTile) {
        auto lhs_block =
            load<RowTile, AccTile, TransposeLHS>(lhs_ptr, row, acc_idx, lhs_ld);
        auto rhs_block =
            load<AccTile, ColTile, TransposeRHS>(rhs_ptr, acc_idx, col, rhs_ld);

        block_mmacc(lhs_block, rhs_block, out_block);
      }

      store_block_unchecked(out_block, out_ptr, out_ld, row, col);
    }
  }

 private:
  ReadAccessor<T const> lhs_;
  ReadAccessor<T const> rhs_;
  ReadWriteAccessor<T> output_;
  Index const batches_;
  Index const m_;
  Index const k_;
  Index const n_;
  T const beta_;
};

}  // namespace matmul
}  // namespace sycldnn
#endif  // SYCLDNN_SRC_MATMUL_KERNELS_H_

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
#ifndef PORTDNN_SRC_MATMUL_BLOCKS_H_
#define PORTDNN_SRC_MATMUL_BLOCKS_H_

#include "src/helpers/math.h"
#include "src/helpers/register_tile.h"
#include "src/helpers/vector_element.h"
#include "src/helpers/vector_io.h"
#include "src/helpers/vector_type.h"

namespace sycldnn {
namespace matmul {

template <typename T, int Rows, int Cols>
struct VectorBlock final
    : public helpers::RegisterTile1D<
          typename helpers::VectorType<T, Cols>::type, Rows> {
  using VectorType = typename helpers::VectorType<T, Cols>::type;
  using helpers::RegisterTile1D<VectorType, Rows>::data;
};

template <typename T, int Rows, int Cols>
static VectorBlock<T, Cols, Rows> SNN_ALWAYS_INLINE
transpose_block(VectorBlock<T, Rows, Cols> const& input) {
  namespace vec_elem = helpers::vector_element;
  VectorBlock<T, Cols, Rows> output;
  SNN_PRAGMA_UNROLL
  for (int i = 0; i < Cols; ++i) {
    SNN_PRAGMA_UNROLL
    for (int j = 0; j < Rows; ++j) {
      vec_elem::set(output.data(i), j, vec_elem::get(input.data(j), i));
    }
  }
  return output;
}

template <typename VectorType, typename T, MULTI_PTR_TEMPLATE_DECL>
static VectorType SNN_ALWAYS_INLINE
load_row(cl::sycl::multi_ptr<T, MULTI_PTR_TEMPLATE> row_start) {
  using VectorLoad = helpers::io::Load<VectorType>;
  VectorType output = VectorLoad()(row_start, 0);
  return output;
}

// if the row is not internal the load_row function is overloaded to accept
// col mask to check the boundary element of the row
template <typename VectorType, int Cols, typename T, MULTI_PTR_TEMPLATE_DECL>
static VectorType SNN_ALWAYS_INLINE
load_row(cl::sycl::multi_ptr<T, MULTI_PTR_TEMPLATE> row_start,
         std::array<bool, Cols> mask) {
  namespace vec_elem = helpers::vector_element;
  using ScalarType = T;
  using ScalarLoad = helpers::io::Load<ScalarType>;
  VectorType output;
  if (mask[Cols - 1]) {
    output = load_row<VectorType>(row_start);
  } else {
    for (int i = 0; i < Cols; ++i) {
      auto val = mask[i] ? ScalarLoad()(row_start, 0) : ScalarType{0};
      vec_elem::set(output, i, val);
      ++row_start;
    }
  }
  return output;
}

// if the block is not internal the load block function is overloaded to accept
// row and col mask to check the boundary
template <int Rows, int Cols, typename T, MULTI_PTR_TEMPLATE_DECL,
          typename Index>
static VectorBlock<T, Rows, Cols> SNN_ALWAYS_INLINE
load_block(cl::sycl::multi_ptr<T const, MULTI_PTR_TEMPLATE> input, Index ld,
           std::array<bool, Rows> row_mask, std::array<bool, Cols> col_mask) {
  using OutputType = VectorBlock<T, Rows, Cols>;
  using VectorType = typename OutputType::VectorType;
  OutputType output;
  auto row_start_ptr = input;
  for (int i = 0; i < Rows; ++i) {
    if (row_mask[i]) {
      output.data(i) = load_row<VectorType, Cols>(row_start_ptr, col_mask);
      row_start_ptr += ld;
    } else {
      output.data(i) = VectorType{0};
    }
  }
  return output;
}

// if the requested is not internal the load function is overloaded to accept
// row and col mask to check the boundary
template <int Rows, int Cols, bool Transpose, typename T,
          MULTI_PTR_TEMPLATE_DECL, typename Index>
static VectorBlock<T, Rows, Cols> SNN_ALWAYS_INLINE
load(cl::sycl::multi_ptr<T const, MULTI_PTR_TEMPLATE> input, Index ld,
     std::array<bool, Rows> row_mask, std::array<bool, Cols> col_mask) {
  VectorBlock<T, Rows, Cols> output;
  if (Transpose) {
    auto out_trans = load_block<Cols, Rows>(input, ld, col_mask, row_mask);
    output = transpose_block(out_trans);
  } else {
    output = load_block<Rows, Cols>(input, ld, row_mask, col_mask);
  }
  return output;
}

template <int Rows, int Cols, typename T, MULTI_PTR_TEMPLATE_DECL,
          typename Index>
static VectorBlock<T, Rows, Cols> SNN_ALWAYS_INLINE
load_block(cl::sycl::multi_ptr<T const, MULTI_PTR_TEMPLATE> input, Index ld) {
  using OutputType = VectorBlock<T, Rows, Cols>;
  using VectorType = typename OutputType::VectorType;
  VectorBlock<T, Rows, Cols> output;
  auto row_start_ptr = input;
  for (int i = 0; i < Rows; ++i) {
    output.data(i) = load_row<VectorType>(row_start_ptr);
    row_start_ptr += ld;
  }
  return output;
}

template <int Rows, int Cols, bool Transpose, typename T,
          MULTI_PTR_TEMPLATE_DECL, typename Index>
static VectorBlock<T, Rows, Cols> SNN_ALWAYS_INLINE
load(cl::sycl::multi_ptr<T const, MULTI_PTR_TEMPLATE> input, Index ld) {
  VectorBlock<T, Rows, Cols> output;
  if (Transpose) {
    auto out_trans = load_block<Cols, Rows>(input, ld);
    output = transpose_block(out_trans);
  } else {
    output = load_block<Rows, Cols>(input, ld);
  }
  return output;
}

template <typename T, int Rows, int Cols>
static void SNN_ALWAYS_INLINE scalar_multiply(VectorBlock<T, Rows, Cols>& block,
                                              T val) {
  using VectorType = typename VectorBlock<T, Rows, Cols>::VectorType;
  VectorType vector_val{val};
  for (int row = 0; row < Rows; ++row) {
    block.data(row) *= vector_val;
  }
}

template <typename T, int Rows, int Cols, int Acc>
static void SNN_ALWAYS_INLINE block_mmacc(
    VectorBlock<T, Rows, Acc> const& lhs, VectorBlock<T, Acc, Cols> const& rhs,
    VectorBlock<T, Rows, Cols>& accumulator) {
  using VectorType = typename VectorBlock<T, Rows, Cols>::VectorType;
  namespace vec_elem = helpers::vector_element;
  for (int row = 0; row < Rows; ++row) {
    for (int acc = 0; acc < Acc; ++acc) {
      accumulator.data(row) =
          helpers::math::mad(VectorType{vec_elem::get(lhs.data(row), acc)},
                             rhs.data(acc), accumulator.data(row));
    }
  }
}

template <int Cols, typename VectorType, typename T, MULTI_PTR_TEMPLATE_DECL>
static void SNN_ALWAYS_INLINE
store_row(VectorType const& row_vec,
          cl::sycl::multi_ptr<T, MULTI_PTR_TEMPLATE> row_start,
          std::array<bool, Cols> valid_col) {
  using ScalarType = T;
  using ScalarStore = helpers::io::Store<ScalarType>;
  namespace vec_elem = helpers::vector_element;
  for (int i = 0; i < Cols; ++i) {
    if (valid_col[i]) {
      ScalarStore()(row_start, 0, vec_elem::get(row_vec, i));
      ++row_start;
    }
  }
}

// if the written block is not internal the store function is overloaded to
// accept row and col mask to check the boundary
template <int Rows, int Cols, typename T, MULTI_PTR_TEMPLATE_DECL,
          typename Index>
static void SNN_ALWAYS_INLINE store_block(
    VectorBlock<T, Rows, Cols> const& block,
    cl::sycl::multi_ptr<T, MULTI_PTR_TEMPLATE> output, Index ld,
    std::array<bool, Rows> valid_row, std::array<bool, Cols> valid_col) {
  auto row_start_ptr = output;
  for (int i = 0; i < Rows; ++i) {
    if (valid_row[i]) {
      store_row<Cols>(block.data(i), row_start_ptr, valid_col);
      row_start_ptr += ld;
    }
  }
}

template <int Rows, int Cols, typename T, MULTI_PTR_TEMPLATE_DECL,
          typename Index>
static void SNN_ALWAYS_INLINE
store_block(VectorBlock<T, Rows, Cols> const& block,
            cl::sycl::multi_ptr<T, MULTI_PTR_TEMPLATE> output, Index ld) {
  using VectorType = typename VectorBlock<T, Rows, Cols>::VectorType;
  using VectorStore = helpers::io::Store<VectorType>;
  for (int i = 0; i < Rows; ++i) {
    VectorStore()(output, 0, block.data(i));
    output += ld;
  }
}

}  // namespace matmul
}  // namespace sycldnn
#endif  // PORTDNN_SRC_MATMUL_BLOCKS

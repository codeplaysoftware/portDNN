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
#ifndef SYCLDNN_SRC_MATMUL_BLOCKS_H_
#define SYCLDNN_SRC_MATMUL_BLOCKS_H_

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

template <typename VectorType, int Cols, bool CheckBounds, typename T,
          cl::sycl::access::address_space Space>
static VectorType SNN_ALWAYS_INLINE
load_row(cl::sycl::multi_ptr<T, Space> row_start, int start_col, int n_cols) {
  namespace vec_elem = helpers::vector_element;
  using ScalarType = T;
  using VectorLoad = helpers::io::Load<VectorType>;
  using ScalarLoad = helpers::io::Load<ScalarType>;
  VectorType output;
  if (CheckBounds) {
    for (int i = 0; i < Cols; ++i) {
      vec_elem::set(output, i,
                    (start_col + i < n_cols ? ScalarLoad()(row_start, i)
                                            : ScalarType{0}));
    }
  } else {
    output = VectorLoad()(row_start, 0);
  }
  return output;
}

template <int Rows, int Cols, bool CheckRows, bool CheckCols, typename T,
          cl::sycl::access::address_space Space>
static VectorBlock<T, Rows, Cols> SNN_ALWAYS_INLINE
load_block(cl::sycl::multi_ptr<T const, Space> input, int start_row,
           int start_col, int ld, int n_rows, int n_cols) {
  using OutputType = VectorBlock<T, Rows, Cols>;
  using VectorType = typename OutputType::VectorType;
  VectorBlock<T, Rows, Cols> output;
  auto row_start_ptr = input + ld * start_row + start_col;
  if (CheckRows) {
    for (int i = 0; i < Rows; ++i) {
      if (start_row + i < n_rows) {
        output.data(i) = load_row<VectorType, Cols, CheckCols>(
            row_start_ptr, start_col, n_cols);
        row_start_ptr += ld;
      } else {
        output.data(i) = VectorType{0};
      }
    }
  } else {
    for (int i = 0; i < Rows; ++i) {
      output.data(i) = load_row<VectorType, Cols, CheckCols>(row_start_ptr,
                                                             start_col, n_cols);
      row_start_ptr += ld;
    }
  }
  return output;
}

template <int Rows, int Cols, typename T, cl::sycl::access::address_space Space>
static VectorBlock<T, Rows, Cols> SNN_ALWAYS_INLINE
load_raw(cl::sycl::multi_ptr<T const, Space> input, int start_row,
         int start_col, int ld, int n_rows, int n_cols) {
  bool check_rows = start_row + Rows >= n_rows;
  bool check_cols = start_col + Cols >= n_cols;
  VectorBlock<T, Rows, Cols> output;
  if (check_rows) {
    if (check_cols) {
      output = load_block<Rows, Cols, true, true>(input, start_row, start_col,
                                                  ld, n_rows, n_cols);
    } else {
      output = load_block<Rows, Cols, true, false>(input, start_row, start_col,
                                                   ld, n_rows, n_cols);
    }
  } else {
    if (check_cols) {
      output = load_block<Rows, Cols, false, true>(input, start_row, start_col,
                                                   ld, n_rows, n_cols);
    } else {
      output = load_block<Rows, Cols, false, false>(input, start_row, start_col,
                                                    ld, n_rows, n_cols);
    }
  }
  return output;
}

template <int Rows, int Cols, bool Transpose, typename T,
          cl::sycl::access::address_space Space>
static VectorBlock<T, Rows, Cols> SNN_ALWAYS_INLINE
load(cl::sycl::multi_ptr<T const, Space> input, int start_row, int start_col,
     int n_rows, int n_cols) {
  VectorBlock<T, Rows, Cols> output;
  if (Transpose) {
    auto out_trans = load_raw<Cols, Rows>(input, start_col, start_row, n_rows,
                                          n_cols, n_rows);
    output = transpose_block(out_trans);
  } else {
    output = load_raw<Rows, Cols>(input, start_row, start_col, n_cols, n_rows,
                                  n_cols);
  }
  return output;
}

template <int Rows, int Cols, bool Transpose, typename T,
          cl::sycl::access::address_space Space>
static VectorBlock<T, Rows, Cols> SNN_ALWAYS_INLINE
load(cl::sycl::multi_ptr<T const, Space> input, int start_row, int start_col,
     int ld) {
  VectorBlock<T, Rows, Cols> output;
  if (Transpose) {
    auto out_trans = load_block<Cols, Rows, false, false>(input, start_col,
                                                          start_row, ld, 0, 0);
    output = transpose_block(out_trans);
  } else {
    output = load_block<Rows, Cols, false, false>(input, start_row, start_col,
                                                  ld, 0, 0);
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

template <int Cols, typename VectorType, typename T,
          cl::sycl::access::address_space Space>
static void SNN_ALWAYS_INLINE store_row(VectorType const& row_vec,
                                        cl::sycl::multi_ptr<T, Space> row_start,
                                        int start_col, int n_cols) {
  using ScalarType = T;
  using ScalarStore = helpers::io::Store<ScalarType>;
  namespace vec_elem = helpers::vector_element;
  for (int i = 0; i < Cols; ++i) {
    if (start_col + i < n_cols) {
      ScalarStore()(row_start, start_col + i, vec_elem::get(row_vec, i));
    }
  }
}

template <int Rows, int Cols, typename T, cl::sycl::access::address_space Space>
static void SNN_ALWAYS_INLINE
store_block(VectorBlock<T, Rows, Cols> const& block,
            cl::sycl::multi_ptr<T, Space> output, int start_row, int start_col,
            int ld, int n_rows, int n_cols) {
  auto row_start_ptr = output + ld * start_row;
  for (int i = 0; i < Rows; ++i) {
    if (start_row + i < n_rows) {
      store_row<Cols>(block.data(i), row_start_ptr, start_col, n_cols);
      row_start_ptr += ld;
    }
  }
}

template <int Rows, int Cols, typename T, cl::sycl::access::address_space Space>
static void SNN_ALWAYS_INLINE
store_block_unchecked(VectorBlock<T, Rows, Cols> const& block,
                      cl::sycl::multi_ptr<T, Space> output, int ld,
                      int start_row, int start_col) {
  using VectorType = typename VectorBlock<T, Rows, Cols>::VectorType;
  using VectorStore = helpers::io::Store<VectorType>;
  auto row_start_ptr = output + ld * start_row + start_col;
  for (int i = 0; i < Rows; ++i) {
    VectorStore()(row_start_ptr, 0, block.data(i));
    row_start_ptr += ld;
  }
}

}  // namespace matmul
}  // namespace sycldnn
#endif  // SYCLDNN_SRC_MATMUL_BLOCKS

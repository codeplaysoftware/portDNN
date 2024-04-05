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
#ifndef SYCLDNN_INCLUDE_INTERNAL_HELPERS_LOCAL_MEM_TILE_H_
#define SYCLDNN_INCLUDE_INTERNAL_HELPERS_LOCAL_MEM_TILE_H_

/**
 * \file
 * Implements the \ref sycldnn::helpers::internal::LocalMemTile struct, which
 * provides functionality to load a window of data from global memory into local
 * memory, transpose and/or reduce the data in local memory and then write back
 * out to global memory.
 *
 */

#include <CL/sycl.hpp>
#include <algorithm>

#include "sycldnn/helpers/macros.h"
#include "sycldnn/helpers/ratio.h"

#include "src/helpers/vector_io.h"

namespace sycldnn {
namespace helpers {
namespace internal {

/**
 * Calculates the max column tile used by the `LocalMemTile` struct, given the
 * max workgroup size and the max subgroup size. The result being min(col_lim,
 * max_wg_size/max_sg_size)
 *
 * \tparam Index Index type
 * \param col_lim The column limit
 * \param max_wg_size The maximum workgroup size of the device
 * \param max_sg_size The maximum subgroup size of the device
 * \return The maximum column tile
 */
template <typename Index>
Index get_max_col_tile_dim(Index col_lim, Index max_wg_size,
                           Index max_sg_size) {
  // We want the tile_row to be at least `max_sg_size`
  auto const max_tile_col = max_wg_size / max_sg_size;
  return std::min(col_lim, max_tile_col);
}

/**
 * Calculates the max row tile used by the `LocalMemTile` struct, given the
 * column tile, row_lim, max workgroup size and max subgroup size.
 *
 * \tparam Index Index type
 * \param row_lim The row limit
 * \param col_tile The column size of the window
 * \param max_wg_size The maximum workgroup size of the device
 * \param max_sg_size The maximum subgroup size of the device
 * \return The maximum row tile
 */
template <typename Index>
Index get_max_row_tile_dim(Index row_lim, Index tile_col, Index max_wg_size,
                           Index max_sg_size) {
  // The max value row_tile can take and still fit in the subgroup
  auto const max_row_tile = max_wg_size / tile_col;
  // Our ideal row tile.
  auto const ideal_row_tile =
      round_up_to_nearest_multiple(row_lim, max_sg_size);
  // Upperbound on our row tile
  auto ub_row_tile = round_up_to_nearest_multiple(max_row_tile, max_sg_size);
  // Nearest multiple might be too large
  ub_row_tile =
      ub_row_tile > max_row_tile ? ub_row_tile - max_sg_size : ub_row_tile;
  return std::min(ideal_row_tile, ub_row_tile);
}

/**
 * \struct LocalMemTile
 * Helper struct used to manage the loading of data from global memory,
 * transposing and/or reducing it in local memory and then storing it back to
 * global memory. It's expected that the kernel is launched with a 3d nd_range
 * where the local range is (1, row_tile, col_tile) and the global range is
 * (batch_size, global_row, global_col). Where `global_row` and `global_col` are
 * multiples of `row_tile` and `col_tile` respectively.
 *
 * \tparam T The data type
 * \tparam Index The index type
 * \tparam Dim The nummber of dimensions of the `cl::sycl::nd_item` passed to
 *         the constructor. (Expected to be 3 for now)
 * \tparam CheckBounds Flag indicating if bounds checks are
 *         needed.
 * \tparam BankOffset Flag indicating if to offset data to avoid bank
 *         conflicts
 */
template <typename T, typename Index, int Dim, bool CheckBounds,
          bool BankOffset = true>
struct LocalMemTile {
  static constexpr Index RowDim = Dim == 3 ? 1 : 0;
  static constexpr Index ColDim = Dim == 3 ? 2 : 1;
  Index row_tile_;
  Index col_tile_;
  Index row_lim_;
  Index col_lim_;
  Index local_id_;
  Index local_row_id_;
  Index local_col_id_;
  Index wg_id_;
  T pad_val_;
  cl::sycl::nd_item<Dim>& nd_item_;

  /**
   * LocalMemTile constructor
   *
   * \param row_tile The size of the window row
   * \param col_tile The size of the window col
   * \param row_lim The size of the row limits
   * \param col_lim The size of the col limits
   * \param nd_item The cl::sycl::nd_item recieved from the device kernel
   * \param pad_val The padding to add to the ends of the windows if the indices
   *                go beyond the row/col limits
   */
  LocalMemTile(Index row_tile, Index col_tile, Index row_lim, Index col_lim,
               cl::sycl::nd_item<Dim>& nd_item, T pad_val = 0)
      : row_tile_{row_tile},
        col_tile_{col_tile},
        row_lim_{row_lim},
        col_lim_{col_lim},
        local_id_{static_cast<Index>(nd_item.get_local_linear_id())},
        local_row_id_{static_cast<Index>(nd_item.get_local_id(RowDim))},
        local_col_id_{static_cast<Index>(nd_item.get_local_id(ColDim))},
        wg_id_{static_cast<Index>(nd_item.get_group(0))},
        pad_val_{pad_val},
        nd_item_(nd_item) {}

  /**
   * \return The row tile member variable
   */
  SNN_ALWAYS_INLINE Index get_row_tile() { return row_tile_; }

  /**
   * \return The col tile member variable
   */
  SNN_ALWAYS_INLINE Index get_col_tile() { return col_tile_; }

  /**
   * Getter function to calculate the linear local id of the current device
   * thread. Requires a template parameter to indicate if to return the regular
   * id or the transposed one.
   * \tparam Flag indicating if data was transposed
   * \return The calculated linear local id
   */
  template <bool IsTransposed = false>
  SNN_ALWAYS_INLINE Index get_local_id() {
    if constexpr (!IsTransposed) {
      return local_row_id_ * (col_tile_ + BankOffset) + local_col_id_;
    } else {
      return local_col_id_ * (row_tile_ + BankOffset) + local_row_id_;
    }
  }

  /**
   * \return The local row id member variable
   */
  SNN_ALWAYS_INLINE Index get_local_row_id() { return local_row_id_; }

  /**
   * \return The local col id member variable
   */
  SNN_ALWAYS_INLINE Index get_local_col_id() { return local_col_id_; }

  /**
   * This is a getter that returns the global index for the batch dimension.
   * \return The batch workgroup member variable
   */
  SNN_ALWAYS_INLINE Index get_wg_id() { return wg_id_; }

  /**
   * \return The global row id from `nd_item`
   */
  SNN_ALWAYS_INLINE Index get_global_row_id() {
    return nd_item_.get_global_id(RowDim);
  }

  /**
   * \return The global col id from `nd_item`
   */
  SNN_ALWAYS_INLINE Index get_global_col_id() {
    return nd_item_.get_global_id(ColDim);
  }

  /**
   * Loads a window of data from global memory using the row/col configurations
   * for the current batch. If the global row/col indices fall outside the
   * row/col limits then the output is filled with padding values for those
   * indices.
   *
   * \tparam InMem Input memory type
   * \tparam OutMem Output memory type
   * \param global_in Pointer to global input memory
   * \param local_out Pointer to local output memory
   */
  template <typename InMem, typename OutMem>
  SNN_ALWAYS_INLINE void load_memory(InMem const& global_in,
                                     OutMem& local_out) {
    auto const local_id = get_local_id();
    auto const wg_batch_id = get_wg_id();
    auto const global_row_id = get_global_row_id();
    auto const global_col_id = get_global_col_id();

    auto const global_id = wg_batch_id * row_lim_ * col_lim_ +
                           global_row_id * col_lim_ + global_col_id;

    if constexpr (CheckBounds) {
      auto const within_bounds =
          global_row_id < row_lim_ && global_col_id < col_lim_;
      auto const val = within_bounds ? global_in[global_id] : pad_val_;
      store_local(local_out, local_id, val);
    } else {
      // maybe do subgroup load
      store_local(local_out, local_id, global_in[global_id]);
    }
    group_barrier(nd_item_.get_group());
  }

  /**
   * Stores the window data from the local input memory to the global output
   * memory
   * \tparam IsReduced Flag indicating if the data in local memory was reduced
   * \tparam IsTransposed Flag indicating if the data in local memory was
   *         transposed
   * \tparam InMem Input memory type
   * \tparam OutMem Output memory
   *         type
   * \param local_in Pointer to local input memory
   * \param global_out Pointer to global output memory
   */
  template <bool IsReduced, bool IsTransposed, typename InMem, typename OutMem>
  SNN_ALWAYS_INLINE void store_memory(InMem const& local_in,
                                      OutMem& global_out) {
    auto const local_id = get_local_id<IsTransposed>();
    auto const wg_batch_id = get_wg_id();
    auto const col_id = get_global_col_id();

    Index row_id;
    Index row_lim;
    Index global_id;
    if constexpr (IsReduced) {
      // The row data was reduced, thus row_lim is assumed to be 1 and the
      // output shape [batch_size, col_lim]
      row_id = get_local_row_id();
      row_lim = get_row_tile();
      global_id = wg_batch_id * col_lim_ + col_id;
    } else {
      // Row data was not reduced so the output is shape
      // [batch_size, row_lim, col_lim]/[batch_size, col_lim, row_lim]
      row_id = get_global_row_id();
      row_lim = row_lim_;
      if constexpr (IsTransposed) {
        global_id = (wg_batch_id * col_lim_ + col_id) * row_lim_ + row_id;
      } else {
        global_id = (wg_batch_id * row_lim_ + row_id) * col_lim_ + col_id;
      }
    }

    if (row_id < row_lim && col_id < col_lim_) {
      global_out[global_id] = local_in[local_id];
    }
  }

  /**
   * \tparam InMem input memory type
   * \param input Local input memory
   * \param index Offset to requested value in local memory
   * \return Returns value from offset local pointer
   */
  template <typename InMem>
  SNN_ALWAYS_INLINE T load_local(InMem const& input, int index) {
    return input[index];
  }

  /**
   * Stores given value at requested position in output local memory
   *
   * \tparam OutMem output memory type
   * \param output Local output memory
   * \param index Offset to requested value in local memory
   * \param value Value to update requested position with
   */
  template <typename OutMem>
  SNN_ALWAYS_INLINE void store_local(OutMem& output, int index, T val) {
    output[index] = val;
  }

 private:
  /**
   * Private member function used to transpose the given input into the given
   * output.
   *
   * \tparam Inplace Flag indicating the transpose is being done inplace.
   * \tparam InMem Input memory type
   * \tparam InMem Output memory type
   * \param input Input pointer
   * \param output Output pointer
   */
  template <bool Inplace, typename InMem, typename OutMem>
  SNN_ALWAYS_INLINE void transpose_impl_(InMem const& input, OutMem& output) {
    auto const local_id = get_local_id<false>();
    auto const trans_local_id = get_local_id<true>();
    auto group = nd_item_.get_group();
    // Load memory into registers
    const auto val = load_local(input, local_id);
    if constexpr (Inplace) {
      // Synchronise to ensure all work items pulled memory into registers first
      // before overwriting inplace.
      group_barrier(group);
    }
    // Store transposed memory.
    store_local(output, trans_local_id, val);
    // Sync to ensure all threads fill the local memory
    group_barrier(group);
  }

 public:
  /**
   * Transposes the windowed data out-of-place. Input can either be the local
   * memory that was loaded using the tile or and offset global pointer.
   *
   * \tparam InMem Input memory type
   * \tparam OutMem Output memory type
   * \param input Input pointer
   * \param output Output pointer
   */
  template <typename InMem, typename OutMem>
  SNN_ALWAYS_INLINE void transpose(InMem const& input, OutMem& output) {
    transpose_impl_<false>(input, output);
  }

  /**
   * Transposes the windowed data in-place. Input can either be the local
   * memory that was loaded using the tile or and offset global pointer.
   *
   * \tparam InMem Input memory type
   * \tparam OutMem Output memory type
   * \param input Input pointer
   * \param output Output pointer
   */
  template <typename InMem>
  SNN_ALWAYS_INLINE void transpose(InMem& input) {
    transpose_impl_<true>(input, input);
  }

  /**
   * Performs a single-batch of subgroup reductions. I.e. will reduce tile from
   * [row_tile, col_tile] to [row_tile/max_sg_size, col_tile]. Function will
   * update the the internal row_tile_ member variable to the new reduced amount
   * and will simply return if `row_tile==1`, thus, it is safe to run in a loop.
   * Operation can also be done in-place if the same pointer is passed for
   * input/output.
   *
   * \tparam ReduceOp The reduction operation
   * \tparam IsTransposed Flag to indicate if the data was transposed
   * \tparam InMem Input memory type
   * \tparam OutMem Output memory type
   * \param input Input pointer
   * \param input Output pointer (Can be the same as input)
   */
  template <typename ReduceOp, bool IsTransposed, typename InMem,
            typename OutMem>
  SNN_ALWAYS_INLINE void sg_reduce(InMem const input, OutMem output) {
    Index const reduce_lim = row_tile_;
    Index const stride_lim = col_tile_;
    if (row_tile_ == 1) {
      return;
    }

    auto op = ReduceOp::Op;
    auto group = nd_item_.get_group();
    auto subgroup = nd_item_.get_sub_group();

    Index const sg_size = subgroup.get_max_local_range()[0];
    Index const sg_group_id = subgroup.get_group_linear_id();
    Index const sg_local_id = subgroup.get_local_linear_id();

    // The tile is read in the shape [row_tile, col_tile] ->
    // [num_row_blocks*max_subgroup_size, col_tile]. Thus the local range has
    // the same shape (num_row_blocks*max_subgroup_size, col_tile), using this
    // we can assign num_row_blocks*col_tile subgroups to reduce a chunk of
    // memory size max_subgroup_size

    // Get the number of reduction blocks each sized [max_subgroup_size]
    auto const num_reduce_blocks =
        cl::sycl::max((reduce_lim) / sg_size, Index{1});
    // Get the id of the reduction block
    auto const reduce_block_id = sg_group_id % num_reduce_blocks;
    // Use the reduction block id to calculate the index in the tile range
    auto const local_reduce_id = reduce_block_id * sg_size + sg_local_id;
    // Get the stride id in the tile range
    auto const local_stride_id = sg_group_id / num_reduce_blocks;

    Index input_id;   // Index of input local memory
    Index output_id;  // Index of reduced data into output local memory
    if constexpr (!IsTransposed) {
      // Reduce accross rows in strided access
      input_id = local_reduce_id * (stride_lim + BankOffset) + local_stride_id;
      // Store in strided access
      output_id = reduce_block_id * (stride_lim + BankOffset) + local_stride_id;
    } else {
      // Reduce accross columns in coalesced access
      input_id = local_stride_id * (reduce_lim + BankOffset) + local_reduce_id;
      // Store in coalesced access
      output_id =
          local_stride_id * (num_reduce_blocks + BankOffset) + reduce_block_id;
    }
    auto const sg_input = local_reduce_id < reduce_lim
                              ? load_local(input, input_id)
                              : ReduceOp::Identity;
    // Perform subgroup reduction
    auto sg_output = sycl::reduce_over_group(subgroup, sg_input, op);
    // Sync to ensure reduction is finished
    group_barrier(group);

    if (subgroup.leader()) {
      // Only store result if the leader of subgroup
      store_local(output, output_id, sg_output);
    }
    // Update row tile to new amount
    row_tile_ = num_reduce_blocks;
    // Sync to ensure results are synced.
    group_barrier(group);
  }

  /**
   * Performs batches of out-of-place subgroup reductions in an unrolled loop.
   * Tile row will be reduced from [row_tile, col_tile] ->
   * [row_tile/(NumSGReductions*max_sg_size), col_tile]. NB: function will
   * return once `tile_row==1`.
   *
   * \tparam ReduceOp The reduction operation
   * \tparam IsTransposed Flag to indicate if the data was transposed
   * \tparam InMem Input memory type
   * \tparam OutMem Output memory type
   * \param input Input pointer
   * \param input Output pointer (Can be the same as input)
   */
  template <typename ReduceOp, bool IsTransposed, int NumSGReductions,
            typename InMem, typename OutMem>
  SNN_ALWAYS_INLINE void sg_reduce(InMem const& input, OutMem& output) {
    SNN_PRAGMA_UNROLL
    for (auto i = 0; i < NumSGReductions; i++) {
      sg_reduce<ReduceOp, IsTransposed>(input, output);
    }
  }

  /**
   * Performs batches of in-place subgroup reductions in an unrolled loop.
   * Tile row will be reduced from [row_tile, col_tile] ->
   * [row_tile/(NumSGReductions*max_sg_size), col_tile].
   * NB: function will return once `tile_row==1`.
   *
   * \tparam ReduceOp The reduction operation
   * \tparam IsTransposed Flag to indicate if the data was transposed
   * \tparam InMem Input memory type
   * \tparam OutMem Output memory type
   * \param input Input pointer
   * \param input Output pointer (Can be the same as input)
   */
  template <typename ReduceOp, bool IsTransposed, int NumSGReductions,
            typename InMem>
  SNN_ALWAYS_INLINE void sg_reduce(InMem& input) {
    SNN_PRAGMA_UNROLL
    for (auto i = 0; i < NumSGReductions; i++) {
      sg_reduce<ReduceOp, IsTransposed>(input, input);
    }
  }

  /**
   * Performs batches of work-item reductions, where the work-items with
   * row_id==0 reduce the rows of the tile. Assumes the output row is filled
   * with the first row of the input.
   *
   * \tparam ReduceOp The reduction operation
   * \tparam IsTransposed Flag to indicate if the data was transposed
   * \tparam InMem Input memory type
   * \tparam OutMem Output memory type
   * \param input Input pointer
   * \param input Output pointer (Can be the same as input)
   */
  template <typename ReduceOp, bool IsTransposed, typename InMem,
            typename OutMem>
  SNN_ALWAYS_INLINE void wi_reduce(InMem const& input, OutMem& output) {
    if (row_lim_ == 1) {
      return;
    }

    auto output_id = get_local_id<IsTransposed>();
    if (get_local_row_id() == 0) {
      auto input_id = output_id;
      for (auto row_id = 1; row_id < row_tile_; row_id++) {
        if constexpr (IsTransposed) {
          input_id++;
        } else {
          input_id += (col_tile_ + BankOffset);
        }
        output[output_id] = ReduceOp::Op(input[input_id], output[output_id]);
      }
    }

    group_barrier(nd_item_.get_group());
    row_tile_ = 1;

    if (get_local_row_id() == 0) {
      auto final_output_id = get_local_col_id() * (row_tile_ + BankOffset);
      output[final_output_id] = output[output_id];
    }
    group_barrier(nd_item_.get_group());
  }

  /**
   * Performs batches of work-item reductions, where the work-items with
   * row_id==0 reduce the rows of the tile.
   *
   * \tparam ReduceOp The reduction operation
   * \tparam IsTransposed Flag to indicate if the data was transposed
   * \tparam InMem Input memory type
   * \param input Input pointer
   */
  template <typename ReduceOp, bool IsTransposed, typename InMem>
  SNN_ALWAYS_INLINE void wi_reduce(InMem& input) {
    wi_reduce<ReduceOp, IsTransposed>(input, input);
  }

  /**
   * Finalizes the reduction
   * \tparam ReduceOp Reduction operator
   * \tparam IsTransposed Flag to indicate if data has been transposed.
   * \tparam InMem Input memory type
   * \param input Input memory
   * \param finalize_param Parameter to pass to finalize member of ReductionOp
   */
  template <typename ReduceOp, bool IsTransposed, typename InMem>
  SNN_ALWAYS_INLINE void finalize_reduce(InMem input, Index finalize_param) {
    if constexpr (ReduceOp::NeedsFinalize) {
      const auto local_row_id = get_local_row_id();
      const auto local_id = get_local_id<IsTransposed>();
      if (local_row_id == 0) {
        ReduceOp::finalize(input[local_id], finalize_param);
      }
    }
  }

  /**
   * Performs atomic reduction to reduce the row of the tile. Will also reduce
   * the batch depending on the batch_groups.
   *
   * \tparam ReduceOp Reduction operator
   * \tparam IsTransposed Flag to indicate if data has been transposed.
   * \tparam Order `cl::sycl::memory_order`
   * \tparam Scope `cl::sycl::memory_scope`
   * \tparam Space `cl::sycl::address_space`
   * \tparam InMem Input memory type
   * \param input Input memory
   * \param output Output memory
   * \param finalize_param Parameter to pass to atomic_op member of ReductionOp
   * \param batch_groups Batch groups wraps the batch index in the range [0,
   *                     batch_groups-1] which allows the batch dimension to
   *                     also be reduced
   */
  template <typename ReduceOp, bool IsTransposed, cl::sycl::memory_order Order,
            cl::sycl::memory_scope Scope, cl::sycl::access::address_space Space,
            typename InMem, typename OutMem>
  SNN_ALWAYS_INLINE void atomic_reduce(InMem const& input, OutMem& output,
                                       Index finalize_param,
                                       Index batch_groups) {
    bool constexpr GlobalReduce =
        Space == cl::sycl::access::address_space::global_space;

    // Get local indexes
    auto const local_row_id = get_local_row_id();
    auto const local_col_id = get_local_col_id();

    // Get global column id and limit
    auto const global_col_id =
        GlobalReduce ? get_global_col_id() : local_col_id;
    auto const global_col_lim = GlobalReduce ? col_lim_ : col_tile_;

    auto const row_id_ok = local_row_id < row_tile_;
    auto const col_id_ok = GlobalReduce ? global_col_id < global_col_lim : true;

    if (row_id_ok && col_id_ok) {
      auto const outer_id = GlobalReduce ? get_wg_id() : 0;
      auto const output_id =
          (outer_id % batch_groups) * global_col_lim + global_col_id;
      auto const input_id = get_local_id<IsTransposed>();
      ReduceOp::template atomic_op<Order, Scope, Space>(
          output[output_id], input[input_id], finalize_param);
    }
    // Atomics do a full reduction of the row.
    row_tile_ = 1;
  }
};

}  // namespace internal
}  // namespace helpers
}  // namespace sycldnn
#endif

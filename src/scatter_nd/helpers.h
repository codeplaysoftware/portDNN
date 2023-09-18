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

#ifndef PORTDNN_SRC_SCATTER_ND_HELPERS_H_
#define PORTDNN_SRC_SCATTER_ND_HELPERS_H_
#include "portdnn/helpers/macros.h"

namespace sycldnn {
namespace scatter_nd {

/**
 * \file
 * Contains the implementation of \ref sycldnn::scatter_nd::IndexHelper which
 * performs the index calculations and checks in the
 * sycldnn::scatter_nd::ScatterNDOp kernel.
 */

/**
 * Helper struct that calculates the offsets needed for flattening the index,
 * given the input tensor dimensions.
 */
template <int IndexDepth>
struct IndexHelper {
  /*@{*/
  size_t dim_0_; /**< First dimension of data tensor*/
  size_t dim_1_; /**< Second dimension of data tensor*/
  size_t dim_2_; /**< Third dimension of data tensor*/
  size_t dim_3_; /**< Fourth dimension of data tensor*/
  int offset_0_; /**< Offset to the first index*/
  int offset_1_; /**< Offset to the second index*/
  int offset_2_; /**< Offset to the third index*/
  /*@}*/
  /**
   * \tparam IndexDepth  The second dimension of the index matrix
   * \param  dim_0       First dimension of data tensor
   * \param  dim_1       Second dimension of data tensor
   * \param  dim_2       Third dimension of data tensor
   * \param  dim_3       Fourth dimension of data tensor
   */
  IndexHelper(int dim_0, int dim_1, int dim_2, int dim_3)
      : dim_0_(static_cast<size_t>(dim_0)),
        dim_1_(static_cast<size_t>(dim_1)),
        dim_2_(static_cast<size_t>(dim_2)),
        dim_3_(static_cast<size_t>(dim_3)),
        offset_0_(dim_1 * dim_2 * dim_3),
        offset_1_(dim_2 * dim_3),
        offset_2_(dim_3) {}
  /**
   * Calculates the output offset using the indices specified in the index data
   * for the specified row of the index matrix. Negative indices are wrapped.
   * Method returns -1 for out of bounds indices.
   *
   * \tparam MultiPtr Device pointer type
   * \tparam Itype Index type
   * \param index_ptr Device pointer to index tensor
   * \param index_row Specified row of the index tensor to get the indices from
   * \return Returns the flattened wrapped index from the specified row in the
   * index tensor. Returns -1 if the index is out of bounds.
   */
  template <typename MultiPtr, typename IType>
  SNN_ALWAYS_INLINE IType operator()(MultiPtr index_ptr, IType index_row) const;

  /**
   * Returns the dimension specified by the template argument
   *
   * \tparam Dim Specified dimension position
   * \return Returns the requested dimension
   */
  template <int Dim>
  SNN_ALWAYS_INLINE int get_dim() const {
    switch (Dim) {
      case 0:
        return dim_0_;
      case 1:
        return dim_1_;
      case 2:
        return dim_2_;
      case 3:
        return dim_3_;
    }
  }
  /**
   * Get the column entry for a specified offset row
   *
   * \tparam IndexCol         Column index of the index tensor row
   * \tparam MultiPtr         Device pointer type
   * \tparam IType            Index type
   * \param index_ptr         Device pointer for index tensor
   * \param index_row_offset  Offset to the desired row in the index tensor
   * \return Returns the desired index at the specified offset position
   */
  template <int IndexCol, typename MultiPtr, typename IType>
  SNN_ALWAYS_INLINE IType get_index(MultiPtr index_ptr,
                                    IType index_row_offset) const {
    return *(index_ptr + index_row_offset + IndexCol);
  }
  /**
   * Checks that the index doesn't fall outside the boundary for the specified
   * dimension.
   *
   * \tparam Dim   The dimension of the data tensor
   * \tparam IType Index type
   * \param idx    The index
   * \return Returns true if the index is within the dimension bounds and false
   * otherwise.
   *
   */
  template <int Dim, typename IType>
  SNN_ALWAYS_INLINE bool bounds_check(IType idx) const {
    IType dim = get_dim<Dim>();
    return idx >= -dim && idx < dim;
  }
  /**
   * Negative indices are wrapped to their positive counterparts. Positive
   * indices remain unchanged.
   *
   * \tparam Dim   The dimension of the data tensor
   * \tparam IType Index type
   * \param idx    The index
   * \return Returns the wrapped index
   */
  template <int Dim, typename IType>
  SNN_ALWAYS_INLINE IType wrap_index(IType idx) const {
    IType dim = get_dim<Dim>();
    return (idx + dim) % dim;
  }
};

/**
 * Macro that checks if the idx is within the bounds of the specified dimension
 * and returns -1 as a failure if not.
 */
#define SNN_BOUNDS_CHECK(Dim, idx) \
  if (!bounds_check<Dim>(idx)) {   \
    return -1;                     \
  }
/**
 * Implementation of the operator() method when IndexDepth=1.
 *
 * \tparam MultiPtr Device pointer type
 * \tparam Itype Index type
 * \param index_ptr Device pointer to index tensor
 * \param index_row Specified row of the index tensor to get the indices from
 * \return Returns the flattened wrapped index from the specified row in the
 * index tensor. Returns -1 if the index is out of bounds.
 */
template <>
template <typename MultiPtr, typename IType>
SNN_ALWAYS_INLINE IType IndexHelper<1>::operator()(MultiPtr index_ptr,
                                                   IType index_row) const {
  IType idx_0 = get_index<0>(index_ptr, index_row);

  SNN_BOUNDS_CHECK(0, idx_0)

  idx_0 = wrap_index<0>(idx_0);

  return idx_0 * offset_0_;
}

/**
 * Implementation of the operator() method when IndexDepth=2.
 *
 * \tparam MultiPtr Device pointer type
 * \tparam Itype Index type
 * \param index_ptr Device pointer to index tensor
 * \param index_row Specified row of the index tensor to get the indices from
 * \return Returns the flattened wrapped index from the specified row in the
 * index tensor. Returns -1 if the index is out of bounds.
 */
template <>
template <typename MultiPtr, typename IType>
SNN_ALWAYS_INLINE IType IndexHelper<2>::operator()(MultiPtr index_ptr,
                                                   IType index_row) const {
  IType index_row_offset = index_row * 2;

  auto idx_0 = get_index<0>(index_ptr, index_row_offset);
  auto idx_1 = get_index<1>(index_ptr, index_row_offset);

  SNN_BOUNDS_CHECK(0, idx_0)
  SNN_BOUNDS_CHECK(1, idx_1)

  idx_0 = wrap_index<0>(idx_0);
  idx_1 = wrap_index<1>(idx_1);

  return idx_0 * offset_0_ + idx_1 * offset_1_;
}

/**
 * Implementation of the operator() method when IndexDepth=3.
 *
 * \tparam MultiPtr Device pointer type
 * \tparam Itype Index type
 * \param index_ptr Device pointer to index tensor
 * \param index_row Specified row of the index tensor to get the indices from
 * \return Returns the flattened wrapped index from the specified row in the
 * index tensor. Returns -1 if the index is out of bounds.
 */
template <>
template <typename MultiPtr, typename IType>
SNN_ALWAYS_INLINE IType IndexHelper<3>::operator()(MultiPtr index_ptr,
                                                   IType index_row) const {
  IType index_row_offset = index_row * 3;

  auto idx_0 = get_index<0>(index_ptr, index_row_offset);
  auto idx_1 = get_index<1>(index_ptr, index_row_offset);
  auto idx_2 = get_index<2>(index_ptr, index_row_offset);

  SNN_BOUNDS_CHECK(0, idx_0)
  SNN_BOUNDS_CHECK(1, idx_1)
  SNN_BOUNDS_CHECK(2, idx_2)

  idx_0 = wrap_index<0>(idx_0);
  idx_1 = wrap_index<1>(idx_1);
  idx_2 = wrap_index<2>(idx_2);

  return idx_0 * offset_0_ + idx_1 * offset_1_ + idx_2 * offset_2_;
}

/**
 * Implementation of the operator() method when IndexDepth=4.
 *
 * \tparam MultiPtr Device pointer type
 * \tparam Itype Index type
 * \param index_ptr Device pointer to index tensor
 * \param index_row Specified row of the index tensor to get the indices from
 * \return Returns the flattened wrapped index from the specified row in the
 * index tensor. Returns -1 if the index is out of bounds.
 */
template <>
template <typename MultiPtr, typename IType>
SNN_ALWAYS_INLINE IType IndexHelper<4>::operator()(MultiPtr index_ptr,
                                                   IType index_row) const {
  IType index_row_offset = index_row * 4;
  auto idx_0 = get_index<0>(index_ptr, index_row_offset);
  auto idx_1 = get_index<1>(index_ptr, index_row_offset);
  auto idx_2 = get_index<2>(index_ptr, index_row_offset);
  auto idx_3 = get_index<3>(index_ptr, index_row_offset);

  SNN_BOUNDS_CHECK(0, idx_0)
  SNN_BOUNDS_CHECK(1, idx_1)
  SNN_BOUNDS_CHECK(2, idx_2)
  SNN_BOUNDS_CHECK(3, idx_3)

  idx_0 = wrap_index<0>(idx_0);
  idx_1 = wrap_index<1>(idx_1);
  idx_2 = wrap_index<2>(idx_2);
  idx_3 = wrap_index<3>(idx_3);

  return idx_0 * offset_0_ + idx_1 * offset_1_ + idx_2 * offset_2_ + idx_3;
}
#undef SNN_BOUNDS_CHECK
}  // namespace scatter_nd
}  // namespace sycldnn
#endif  // PORTDNN_SRC_SCATTER_ND_HELPERS_H_

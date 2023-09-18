/*
 * Copyright Codeplay Software Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use these files except in compliance with the License.
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
#ifndef PORTDNN_SRC_HELPERS_TENSOR_INDEX_H_
#define PORTDNN_SRC_HELPERS_TENSOR_INDEX_H_

#include "portdnn/helpers/macros.h"
#include "src/helpers/fast_div.h"

namespace sycldnn {
namespace helpers {
/**
 * A 2D tensor index. The most packed index is s1, with s0 the least packed.
 */
template <typename Index>
struct TensorIndex2D {
  Index s0;
  Index s1;
};
/**
 * A 3D tensor index. The most packed index is s2, with s0 the least packed.
 */
template <typename Index>
struct TensorIndex3D {
  Index s0;
  Index s1;
  Index s2;
};
/**
 * A 4D tensor index. The most packed index is s3, with s0 the least packed.
 */
template <typename Index>
struct TensorIndex4D {
  Index s0;
  Index s1;
  Index s2;
  Index s3;
};
/**
 * Helper class to provide factory methods for the TensorIndex objects from a
 * flattened index. If UseFastDiv is true then the fast division type is used to
 * convert each integer division to a multiply and shift, otherwise a standard
 * integer division is used.
 */
template <typename Index, bool UseFastDiv>
struct TensorIndexHelper {
  using FastDivType = typename fast_div::IndexDiv<Index, UseFastDiv>::type;
  /**
   * Compute a 2D tensor index from a flattened index. The most packed dimension
   * in memory is assumed to be the last one (i.e. the dimension with size_1
   * elements), while the size of the least packed dimension is not needed for
   * the calculation.
   */
  static SNN_ALWAYS_INLINE TensorIndex2D<Index> unflatten2d(
      Index index, FastDivType div_size_1, Index size_1) {
    Index const s01_idx = index;
    Index const s0 = s01_idx / div_size_1;
    Index const s1 = s01_idx - s0 * size_1;

    TensorIndex2D<Index> result{s0, s1};
    return result;
  }
  /**
   * Compute a 3D tensor index from a flattened index. The most packed dimension
   * in memory is assumed to be the last one (i.e. the dimension with size_2
   * elements), while the size of the least packed dimension is not needed for
   * the calculation.
   */
  static SNN_ALWAYS_INLINE TensorIndex3D<Index> unflatten3d(
      Index index, FastDivType div_size_1, Index size_1, FastDivType div_size_2,
      Index size_2) {
    Index const s012_idx = index;
    Index const s01_idx = s012_idx / div_size_2;
    Index const s2 = s012_idx - s01_idx * size_2;
    Index const s0 = s01_idx / div_size_1;
    Index const s1 = s01_idx - s0 * size_1;

    TensorIndex3D<Index> result{s0, s1, s2};
    return result;
  }
  /**
   * Compute a 4D tensor index from a flattened index. The most packed dimension
   * in memory is assumed to be the last one (i.e. the dimension with size_3
   * elements), while the size of the least packed dimension is not needed for
   * the calculation.
   */
  static SNN_ALWAYS_INLINE TensorIndex4D<Index> unflatten4d(
      Index index, FastDivType div_size_1, Index size_1, FastDivType div_size_2,
      Index size_2, FastDivType div_size_3, Index size_3) {
    Index const s0123_idx = index;
    Index const s012_idx = s0123_idx / div_size_3;
    Index const s3 = s0123_idx - s012_idx * size_3;
    Index const s01_idx = s012_idx / div_size_2;
    Index const s2 = s012_idx - s01_idx * size_2;
    Index const s0 = s01_idx / div_size_1;
    Index const s1 = s01_idx - s0 * size_1;

    TensorIndex4D<Index> result{s0, s1, s2, s3};
    return result;
  }
};
/**
 * When not using fast integer divisions, we can ignore the fast_div variable.
 * This means the fast_div variable will not be used and so can be removed as a
 * kernel parameter by the SYCL device compiler.
 */
template <typename Index>
struct TensorIndexHelper<Index, false> {
  using FastDivType = typename fast_div::IndexDiv<Index, false>::type;
  static SNN_ALWAYS_INLINE TensorIndex2D<Index> unflatten2d(
      Index index, FastDivType /*div_size_1*/, Index size_1) {
    Index const s01_idx = index;
    Index const s0 = s01_idx / size_1;
    Index const s1 = s01_idx % size_1;

    TensorIndex2D<Index> result{s0, s1};
    return result;
  }
  static SNN_ALWAYS_INLINE TensorIndex3D<Index> unflatten3d(
      Index index, FastDivType /*div_size_1*/, Index size_1,
      FastDivType /*div_size_2*/, Index size_2) {
    Index const s012_idx = index;
    Index const s01_idx = s012_idx / size_2;
    Index const s2 = s012_idx % size_2;
    Index const s0 = s01_idx / size_1;
    Index const s1 = s01_idx % size_1;

    TensorIndex3D<Index> result{s0, s1, s2};
    return result;
  }
  static SNN_ALWAYS_INLINE TensorIndex4D<Index> unflatten4d(
      Index index, FastDivType /*div_size_1*/, Index size_1,
      FastDivType /*div_size_2*/, Index size_2, FastDivType /*div_size_3*/,
      Index size_3) {
    Index const s0123_idx = index;
    Index const s012_idx = s0123_idx / size_3;
    Index const s3 = s0123_idx % size_3;
    Index const s01_idx = s012_idx / size_2;
    Index const s2 = s012_idx % size_2;
    Index const s0 = s01_idx / size_1;
    Index const s1 = s01_idx % size_1;

    TensorIndex4D<Index> result{s0, s1, s2, s3};
    return result;
  }
};
}  // namespace helpers
}  // namespace sycldnn
#endif  // PORTDNN_SRC_HELPERS_TENSOR_INDEX_H_

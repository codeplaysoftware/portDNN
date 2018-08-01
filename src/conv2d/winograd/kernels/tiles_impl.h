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
#ifndef SYCLDNN_SRC_CONV2D_WINOGRAD_KERNELS_TILES_IMPL_H_
#define SYCLDNN_SRC_CONV2D_WINOGRAD_KERNELS_TILES_IMPL_H_

#include "src/conv2d/winograd/kernels/tiles.h"

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace winograd {

/**
 * For any Winograd tiling you wish to implement, ensure that the following
 * three specialisations are completed:
 * \code
 * template <typename T>
 * struct TransformedFilterTile<T, 2, 2, 3, 3> final
 *     : public BaseTransformedFilterTile<T, 2, 2, 3, 3> {
 *   using BaseTransformedFilterTile<T, 2, 2, 3, 3>::data;
 *
 *   template <typename ConvType>
 *   SNN_ALWAYS_INLINE TransformedFilterTile(
 *       FilterTile<T, 2, 2, 3, 3, ConvType> const& filter)
 *       : BaseTransformedFilterTile<T, 2, 2, 3, 3>{} {
 *     // ...
 *   }
 * };
 *
 * template <typename T>
 * struct TransformedInputTile<T, 2, 2, 3, 3> final
 *     : public BaseTransformedInputTile<T, 2, 2, 3, 3> {
 *   using BaseTransformedInputTile<T, 2, 2, 3, 3>::data;
 *
 *   SNN_ALWAYS_INLINE TransformedInputTile(
 *       InputTile<T, 2, 2, 3, 3> const& inp)
 *       : BaseTransformedInputTile<T, 2, 2, 3, 3>{} {
 *     // ...
 *   }
 * };
 *
 * template <typename T>
 * struct OutputTile<T, 2, 2, 3, 3> final : public BaseOutputTile<T, 2, 2, 3, 3>
 * {
 *   using BaseOutputTile<T, 2, 2, 3, 3>::data;
 *   SNN_ALWAYS_INLINE OutputTile(
 *       IntermediateTile<T, 2, 2, 3, 3> const& tile)
 *       : BaseOutputTile<T, 2, 2, 3, 3>{} {
 *     // ...
 *   }
 * };
 * \endcode
 */
template <typename T>
struct TransformedFilterTile<T, 2, 2, 3, 3> final
    : public BaseTransformedFilterTile<T, 2, 2, 3, 3> {
  using BaseTransformedFilterTile<T, 2, 2, 3, 3>::data;
  /**
   * Apply the Winograd transform to the filter tile.
   */
  template <typename ConvType>
  SNN_ALWAYS_INLINE TransformedFilterTile(
      FilterTile<T, 2, 2, 3, 3, ConvType> const& filter)
      : BaseTransformedFilterTile<T, 2, 2, 3, 3>{} {
    data[0][0] = filter.data[0][0];
    data[0][1] =
        (filter.data[0][0] + filter.data[0][1] + filter.data[0][2]) / 2;
    data[0][2] =
        (filter.data[0][0] - filter.data[0][1] + filter.data[0][2]) / 2;
    data[0][3] = filter.data[0][2];

    data[1][0] =
        (filter.data[0][0] + filter.data[1][0] + filter.data[2][0]) / 2;
    data[1][1] = (filter.data[0][0] + filter.data[0][1] + filter.data[0][2] +
                  filter.data[1][0] + filter.data[1][1] + filter.data[1][2] +
                  filter.data[2][0] + filter.data[2][1] + filter.data[2][2]) /
                 4;
    data[1][2] = (filter.data[0][0] - filter.data[0][1] + filter.data[0][2] +
                  filter.data[1][0] - filter.data[1][1] + filter.data[1][2] +
                  filter.data[2][0] - filter.data[2][1] + filter.data[2][2]) /
                 4;
    data[1][3] =
        (filter.data[0][2] + filter.data[1][2] + filter.data[2][2]) / 2;

    data[2][0] =
        (filter.data[0][0] - filter.data[1][0] + filter.data[2][0]) / 2;
    data[2][1] = (filter.data[0][0] + filter.data[0][1] + filter.data[0][2] -
                  filter.data[1][0] - filter.data[1][1] - filter.data[1][2] +
                  filter.data[2][0] + filter.data[2][1] + filter.data[2][2]) /
                 4;
    data[2][2] = (filter.data[0][0] - filter.data[0][1] + filter.data[0][2] -
                  filter.data[1][0] + filter.data[1][1] - filter.data[1][2] +
                  filter.data[2][0] - filter.data[2][1] + filter.data[2][2]) /
                 4;
    data[2][3] =
        (filter.data[0][2] - filter.data[1][2] + filter.data[2][2]) / 2;

    data[3][0] = filter.data[2][0];
    data[3][1] =
        (filter.data[2][0] + filter.data[2][1] + filter.data[2][2]) / 2;
    data[3][2] =
        (filter.data[2][0] - filter.data[2][1] + filter.data[2][2]) / 2;
    data[3][3] = filter.data[2][2];
  }
};

template <typename T>
struct TransformedInputTile<T, 2, 2, 3, 3> final
    : public BaseTransformedInputTile<T, 2, 2, 3, 3> {
  using BaseTransformedInputTile<T, 2, 2, 3, 3>::data;
  /**
   * Apply the Winograd transform to the filter tile.
   */
  SNN_ALWAYS_INLINE TransformedInputTile(InputTile<T, 2, 2, 3, 3> const& inp)
      : BaseTransformedInputTile<T, 2, 2, 3, 3>{} {
    data[0][0] =
        inp.data[0][0] + inp.data[2][2] - inp.data[0][2] - inp.data[2][0];
    data[0][1] =
        inp.data[0][1] + inp.data[0][2] - inp.data[2][1] - inp.data[2][2];
    data[0][2] =
        inp.data[0][2] + inp.data[2][1] - inp.data[0][1] - inp.data[2][2];
    data[0][3] =
        inp.data[0][3] + inp.data[2][1] - inp.data[0][1] - inp.data[2][3];

    data[1][0] =
        inp.data[1][0] + inp.data[2][0] - inp.data[1][2] - inp.data[2][2];
    data[1][1] =
        inp.data[1][1] + inp.data[1][2] + inp.data[2][1] + inp.data[2][2];
    data[1][2] =
        inp.data[1][2] + inp.data[2][2] - inp.data[1][1] - inp.data[2][1];
    data[1][3] =
        inp.data[1][3] + inp.data[2][3] - inp.data[1][1] - inp.data[2][1];

    data[2][0] =
        inp.data[1][2] + inp.data[2][0] - inp.data[1][0] - inp.data[2][2];
    data[2][1] =
        inp.data[2][1] + inp.data[2][2] - inp.data[1][1] - inp.data[1][2];
    data[2][2] =
        inp.data[1][1] + inp.data[2][2] - inp.data[1][2] - inp.data[2][1];
    data[2][3] =
        inp.data[1][1] + inp.data[2][3] - inp.data[1][3] - inp.data[2][1];

    data[3][0] =
        inp.data[1][2] + inp.data[3][0] - inp.data[1][0] - inp.data[3][2];
    data[3][1] =
        inp.data[3][1] + inp.data[3][2] - inp.data[1][1] - inp.data[1][2];
    data[3][2] =
        inp.data[1][1] + inp.data[3][2] - inp.data[1][2] - inp.data[3][1];
    data[3][3] =
        inp.data[1][1] + inp.data[3][3] - inp.data[1][3] - inp.data[3][1];
  }
};

template <typename T>
struct OutputTile<T, 2, 2, 3, 3> final : public BaseOutputTile<T, 2, 2, 3, 3> {
  using BaseOutputTile<T, 2, 2, 3, 3>::data;
  /**
   * Apply the Winograd transform to the intermediate tile to give the final
   * output tile.
   */
  SNN_ALWAYS_INLINE OutputTile(IntermediateTile<T, 2, 2, 3, 3> const& tile)
      : BaseOutputTile<T, 2, 2, 3, 3>{} {
    data[0][0] = tile.data[0][0] + tile.data[0][1] + tile.data[0][2] +
                 tile.data[1][0] + tile.data[1][1] + tile.data[1][2] +
                 tile.data[2][0] + tile.data[2][1] + tile.data[2][2];
    data[0][1] = tile.data[0][1] - tile.data[0][2] + tile.data[0][3] +
                 tile.data[1][1] - tile.data[1][2] + tile.data[1][3] +
                 tile.data[2][1] - tile.data[2][2] + tile.data[2][3];
    data[1][0] = tile.data[1][0] + tile.data[1][1] + tile.data[1][2] -
                 tile.data[2][0] - tile.data[2][1] - tile.data[2][2] +
                 tile.data[3][0] + tile.data[3][1] + tile.data[3][2];
    data[1][1] = tile.data[1][1] - tile.data[1][2] + tile.data[1][3] -
                 tile.data[2][1] + tile.data[2][2] - tile.data[2][3] +
                 tile.data[3][1] - tile.data[3][2] + tile.data[3][3];
  }
};

template <typename T>
struct TransformedFilterTile<T, 2, 1, 3, 1> final
    : public BaseTransformedFilterTile<T, 2, 1, 3, 1> {
  using BaseTransformedFilterTile<T, 2, 1, 3, 1>::data;

  template <typename ConvType>
  SNN_ALWAYS_INLINE TransformedFilterTile(
      FilterTile<T, 2, 1, 3, 1, ConvType> const& filter)
      : BaseTransformedFilterTile<T, 2, 1, 3, 1>{} {
    data[0][0] = filter.data[0][0];
    data[1][0] =
        (filter.data[0][0] + filter.data[1][0] + filter.data[2][0]) / 2;
    data[2][0] =
        (filter.data[0][0] - filter.data[1][0] + filter.data[2][0]) / 2;
    data[3][0] = filter.data[2][0];
  }
};

template <typename T>
struct TransformedInputTile<T, 2, 1, 3, 1> final
    : public BaseTransformedInputTile<T, 2, 1, 3, 1> {
  using BaseTransformedInputTile<T, 2, 1, 3, 1>::data;

  SNN_ALWAYS_INLINE TransformedInputTile(InputTile<T, 2, 1, 3, 1> const& inp)
      : BaseTransformedInputTile<T, 2, 1, 3, 1>{} {
    data[0][0] = inp.data[0][0] - inp.data[2][0];
    data[1][0] = inp.data[1][0] + inp.data[2][0];
    data[2][0] = inp.data[2][0] - inp.data[1][0];
    data[3][0] = inp.data[3][0] - inp.data[1][0];
  }
};

template <typename T>
struct OutputTile<T, 2, 1, 3, 1> final : public BaseOutputTile<T, 2, 1, 3, 1> {
  using BaseOutputTile<T, 2, 1, 3, 1>::data;
  SNN_ALWAYS_INLINE OutputTile(IntermediateTile<T, 2, 1, 3, 1> const& tile)
      : BaseOutputTile<T, 2, 1, 3, 1>{} {
    data[0][0] = tile.data[0][0] + tile.data[1][0] + tile.data[2][0];
    data[1][0] = tile.data[1][0] - tile.data[2][0] + tile.data[3][0];
  }
};

template <typename T>
struct TransformedFilterTile<T, 1, 2, 1, 3> final
    : public BaseTransformedFilterTile<T, 1, 2, 1, 3> {
  using BaseTransformedFilterTile<T, 1, 2, 1, 3>::data;

  template <typename ConvType>
  SNN_ALWAYS_INLINE TransformedFilterTile(
      FilterTile<T, 1, 2, 1, 3, ConvType> const& filter)
      : BaseTransformedFilterTile<T, 1, 2, 1, 3>{} {
    data[0][0] = filter.data[0][0];
    data[0][1] =
        (filter.data[0][0] + filter.data[0][1] + filter.data[0][2]) / 2;
    data[0][2] =
        (filter.data[0][0] - filter.data[0][1] + filter.data[0][2]) / 2;
    data[0][3] = filter.data[0][2];
  }
};

template <typename T>
struct TransformedInputTile<T, 1, 2, 1, 3> final
    : public BaseTransformedInputTile<T, 1, 2, 1, 3> {
  using BaseTransformedInputTile<T, 1, 2, 1, 3>::data;

  SNN_ALWAYS_INLINE TransformedInputTile(InputTile<T, 1, 2, 1, 3> const& inp)
      : BaseTransformedInputTile<T, 1, 2, 1, 3>{} {
    data[0][0] = inp.data[0][0] - inp.data[0][2];
    data[0][1] = inp.data[0][1] + inp.data[0][2];
    data[0][2] = inp.data[0][2] - inp.data[0][1];
    data[0][3] = inp.data[0][3] - inp.data[0][1];
  }
};

template <typename T>
struct OutputTile<T, 1, 2, 1, 3> final : public BaseOutputTile<T, 1, 2, 1, 3> {
  using BaseOutputTile<T, 1, 2, 1, 3>::data;
  SNN_ALWAYS_INLINE OutputTile(IntermediateTile<T, 1, 2, 1, 3> const& tile)
      : BaseOutputTile<T, 1, 2, 1, 3>{} {
    data[0][0] = tile.data[0][0] + tile.data[0][1] + tile.data[0][2];
    data[0][1] = tile.data[0][1] - tile.data[0][2] + tile.data[0][3];
  }
};

template <typename T>
struct TransformedFilterTile<T, 3, 1, 2, 1> final
    : public BaseTransformedFilterTile<T, 3, 1, 2, 1> {
  using BaseTransformedFilterTile<T, 3, 1, 2, 1>::data;

  template <typename ConvType>
  SNN_ALWAYS_INLINE TransformedFilterTile(
      FilterTile<T, 3, 1, 2, 1, ConvType> const& filter)
      : BaseTransformedFilterTile<T, 3, 1, 2, 1>{} {
    data[0][0] = filter.data[0][0];
    data[1][0] = (filter.data[0][0] + filter.data[1][0]) / 2;
    data[2][0] = (filter.data[0][0] - filter.data[1][0]) / 2;
    data[3][0] = filter.data[1][0];
  }
};

template <typename T>
struct TransformedInputTile<T, 3, 1, 2, 1> final
    : public BaseTransformedInputTile<T, 3, 1, 2, 1> {
  using BaseTransformedInputTile<T, 3, 1, 2, 1>::data;

  SNN_ALWAYS_INLINE TransformedInputTile(InputTile<T, 3, 1, 2, 1> const& inp)
      : BaseTransformedInputTile<T, 3, 1, 2, 1>{} {
    data[0][0] = inp.data[0][0] - inp.data[2][0];
    data[1][0] = inp.data[1][0] + inp.data[2][0];
    data[2][0] = -inp.data[1][0] + inp.data[2][0];
    data[3][0] = -inp.data[1][0] + inp.data[3][0];
  }
};

template <typename T>
struct OutputTile<T, 3, 1, 2, 1> final : public BaseOutputTile<T, 3, 1, 2, 1> {
  using BaseOutputTile<T, 3, 1, 2, 1>::data;
  SNN_ALWAYS_INLINE OutputTile(IntermediateTile<T, 3, 1, 2, 1> const& tile)
      : BaseOutputTile<T, 3, 1, 2, 1>{} {
    data[0][0] = tile.data[0][0] + tile.data[1][0] + tile.data[2][0];
    data[1][0] = tile.data[1][0] - tile.data[2][0];
    data[2][0] = tile.data[1][0] + tile.data[2][0] + tile.data[3][0];
  }
};

template <typename T>
struct TransformedFilterTile<T, 1, 3, 1, 2> final
    : public BaseTransformedFilterTile<T, 1, 3, 1, 2> {
  using BaseTransformedFilterTile<T, 1, 3, 1, 2>::data;

  template <typename ConvType>
  SNN_ALWAYS_INLINE TransformedFilterTile(
      FilterTile<T, 1, 3, 1, 2, ConvType> const& filter)
      : BaseTransformedFilterTile<T, 1, 3, 1, 2>{} {
    data[0][0] = filter.data[0][0];
    data[0][1] = (filter.data[0][0] + filter.data[0][1]) / 2;
    data[0][2] = (filter.data[0][0] - filter.data[0][1]) / 2;
    data[0][3] = filter.data[0][1];
  }
};

template <typename T>
struct TransformedInputTile<T, 1, 3, 1, 2> final
    : public BaseTransformedInputTile<T, 1, 3, 1, 2> {
  using BaseTransformedInputTile<T, 1, 3, 1, 2>::data;

  SNN_ALWAYS_INLINE TransformedInputTile(InputTile<T, 1, 3, 1, 2> const& inp)
      : BaseTransformedInputTile<T, 1, 3, 1, 2>{} {
    data[0][0] = inp.data[0][0] - inp.data[0][2];
    data[0][1] = inp.data[0][1] + inp.data[0][2];
    data[0][2] = -inp.data[0][1] + inp.data[0][2];
    data[0][3] = -inp.data[0][1] + inp.data[0][3];
  }
};

template <typename T>
struct OutputTile<T, 1, 3, 1, 2> final : public BaseOutputTile<T, 1, 3, 1, 2> {
  using BaseOutputTile<T, 1, 3, 1, 2>::data;
  SNN_ALWAYS_INLINE OutputTile(IntermediateTile<T, 1, 3, 1, 2> const& tile)
      : BaseOutputTile<T, 1, 3, 1, 2>{} {
    data[0][0] = tile.data[0][0] + tile.data[0][1] + tile.data[0][2];
    data[0][1] = tile.data[0][1] - tile.data[0][2];
    data[0][2] = tile.data[0][1] + tile.data[0][2] + tile.data[0][3];
  }
};

template <typename T>
struct TransformedFilterTile<T, 3, 3, 2, 2> final
    : public BaseTransformedFilterTile<T, 3, 3, 2, 2> {
  using BaseTransformedFilterTile<T, 3, 3, 2, 2>::data;

  template <typename ConvType>
  SNN_ALWAYS_INLINE TransformedFilterTile(
      FilterTile<T, 3, 3, 2, 2, ConvType> const& filter)
      : BaseTransformedFilterTile<T, 3, 3, 2, 2>{} {
    data[0][0] = filter.data[0][0];
    data[0][1] = (filter.data[0][0] + filter.data[0][1]) / 2;
    data[0][2] = (filter.data[0][0] - filter.data[0][1]) / 2;
    data[0][3] = filter.data[0][1];

    data[1][0] = (filter.data[0][0] + filter.data[1][0]) / 2;
    data[1][1] = (filter.data[0][0] + filter.data[0][1] + filter.data[1][0] +
                  filter.data[1][1]) /
                 4;
    data[1][2] = (filter.data[0][0] - filter.data[0][1] + filter.data[1][0] -
                  filter.data[1][1]) /
                 4;
    data[1][3] = (filter.data[0][1] + filter.data[1][1]) / 2;

    data[2][0] = (filter.data[0][0] - filter.data[1][0]) / 2;
    data[2][1] = (filter.data[0][0] + filter.data[0][1] - filter.data[1][0] -
                  filter.data[1][1]) /
                 4;
    data[2][2] = (filter.data[0][0] - filter.data[0][1] - filter.data[1][0] +
                  filter.data[1][1]) /
                 4;
    data[2][3] = (filter.data[0][1] - filter.data[1][1]) / 2;

    data[3][0] = filter.data[1][0];
    data[3][1] = (filter.data[1][0] + filter.data[1][1]) / 2;
    data[3][2] = (filter.data[1][0] - filter.data[1][1]) / 2;
    data[3][3] = filter.data[1][1];
  }
};

template <typename T>
struct TransformedInputTile<T, 3, 3, 2, 2> final
    : public BaseTransformedInputTile<T, 3, 3, 2, 2> {
  using BaseTransformedInputTile<T, 3, 3, 2, 2>::data;

  SNN_ALWAYS_INLINE TransformedInputTile(InputTile<T, 3, 3, 2, 2> const& inp)
      : BaseTransformedInputTile<T, 3, 3, 2, 2>{} {
    data[0][0] =
        inp.data[0][0] - inp.data[0][2] - inp.data[2][0] + inp.data[2][2];
    data[0][1] =
        inp.data[0][1] + inp.data[0][2] - inp.data[2][1] - inp.data[2][2];
    data[0][2] =
        -inp.data[0][1] + inp.data[0][2] + inp.data[2][1] - inp.data[2][2];
    data[0][3] =
        -inp.data[0][1] + inp.data[0][3] + inp.data[2][1] - inp.data[2][3];

    data[1][0] =
        inp.data[1][0] - inp.data[1][2] + inp.data[2][0] - inp.data[2][2];
    data[1][1] =
        inp.data[1][1] + inp.data[1][2] + inp.data[2][1] + inp.data[2][2];
    data[1][2] =
        -inp.data[1][1] + inp.data[1][2] - inp.data[2][1] + inp.data[2][2];
    data[1][3] =
        -inp.data[1][1] + inp.data[1][3] - inp.data[2][1] + inp.data[2][3];

    data[2][0] =
        -inp.data[1][0] + inp.data[1][2] + inp.data[2][0] - inp.data[2][2];
    data[2][1] =
        -inp.data[1][1] - inp.data[1][2] + inp.data[2][1] + inp.data[2][2];
    data[2][2] =
        inp.data[1][1] - inp.data[1][2] - inp.data[2][1] + inp.data[2][2];
    data[2][3] =
        inp.data[1][1] - inp.data[1][3] - inp.data[2][1] + inp.data[2][3];

    data[3][0] =
        -inp.data[1][0] + inp.data[1][2] + inp.data[3][0] - inp.data[3][2];
    data[3][1] =
        -inp.data[1][1] - inp.data[1][2] + inp.data[3][1] + inp.data[3][2];
    data[3][2] =
        inp.data[1][1] - inp.data[1][2] - inp.data[3][1] + inp.data[3][2];
    data[3][3] =
        inp.data[1][1] - inp.data[1][3] - inp.data[3][1] + inp.data[3][3];
  }
};

template <typename T>
struct OutputTile<T, 3, 3, 2, 2> final : public BaseOutputTile<T, 3, 3, 2, 2> {
  using BaseOutputTile<T, 3, 3, 2, 2>::data;
  SNN_ALWAYS_INLINE OutputTile(IntermediateTile<T, 3, 3, 2, 2> const& tile)
      : BaseOutputTile<T, 3, 3, 2, 2>{} {
    data[0][0] = tile.data[0][0] + tile.data[0][1] + tile.data[0][2] +
                 tile.data[1][0] + tile.data[1][1] + tile.data[1][2] +
                 tile.data[2][0] + tile.data[2][1] + tile.data[2][2];
    data[0][1] = tile.data[0][1] - tile.data[0][2] + tile.data[1][1] -
                 tile.data[1][2] + tile.data[2][1] - tile.data[2][2];
    data[0][2] = tile.data[0][1] + tile.data[0][2] + tile.data[0][3] +
                 tile.data[1][1] + tile.data[1][2] + tile.data[1][3] +
                 tile.data[2][1] + tile.data[2][2] + tile.data[2][3];

    data[1][0] = tile.data[1][0] + tile.data[1][1] + tile.data[1][2] -
                 tile.data[2][0] - tile.data[2][1] - tile.data[2][2];
    data[1][1] =
        tile.data[1][1] - tile.data[1][2] - tile.data[2][1] + tile.data[2][2];
    data[1][2] = tile.data[1][1] + tile.data[1][2] + tile.data[1][3] -
                 tile.data[2][1] - tile.data[2][2] - tile.data[2][3];

    data[2][0] = tile.data[1][0] + tile.data[1][1] + tile.data[1][2] +
                 tile.data[2][0] + tile.data[2][1] + tile.data[2][2] +
                 tile.data[3][0] + tile.data[3][1] + tile.data[3][2];
    data[2][1] = tile.data[1][1] - tile.data[1][2] + tile.data[2][1] -
                 tile.data[2][2] + tile.data[3][1] - tile.data[3][2];
    data[2][2] = tile.data[1][1] + tile.data[1][2] + tile.data[1][3] +
                 tile.data[2][1] + tile.data[2][2] + tile.data[2][3] +
                 tile.data[3][1] + tile.data[3][2] + tile.data[3][3];
  }
};

}  // namespace winograd
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_CONV2D_WINOGRAD_KERNELS_TILES_IMPL_H_

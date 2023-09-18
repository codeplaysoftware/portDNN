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
#ifndef PORTDNN_SRC_CONV2D_WINOGRAD_KERNELS_TILES_IMPL_H_
#define PORTDNN_SRC_CONV2D_WINOGRAD_KERNELS_TILES_IMPL_H_

#include "src/conv2d/winograd/kernels/tiles.h"

#include "src/helpers/math.h"
#include "src/helpers/register_tile.h"

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
 *   SNN_ALWAYS_INLINE explicit TransformedFilterTile(
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
 *   SNN_ALWAYS_INLINE explicit TransformedInputTile(
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
 *   SNN_ALWAYS_INLINE explicit OutputTile(
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
  SNN_ALWAYS_INLINE explicit TransformedFilterTile(
      FilterTile<T, 2, 2, 3, 3, ConvType> const& filter)
      : BaseTransformedFilterTile<T, 2, 2, 3, 3>{} {
    data(0, 0) = filter.data(0, 0);
    data(0, 1) = helpers::math::ratio(
        filter.data(0, 0) + filter.data(0, 1) + filter.data(0, 2), 2);
    data(0, 2) = helpers::math::ratio(
        filter.data(0, 0) - filter.data(0, 1) + filter.data(0, 2), 2);
    data(0, 3) = filter.data(0, 2);

    data(1, 0) = helpers::math::ratio(
        filter.data(0, 0) + filter.data(1, 0) + filter.data(2, 0), 2);
    data(1, 1) = helpers::math::ratio(
        filter.data(0, 0) + filter.data(0, 1) + filter.data(0, 2) +
            filter.data(1, 0) + filter.data(1, 1) + filter.data(1, 2) +
            filter.data(2, 0) + filter.data(2, 1) + filter.data(2, 2),
        4);
    data(1, 2) = helpers::math::ratio(
        filter.data(0, 0) - filter.data(0, 1) + filter.data(0, 2) +
            filter.data(1, 0) - filter.data(1, 1) + filter.data(1, 2) +
            filter.data(2, 0) - filter.data(2, 1) + filter.data(2, 2),
        4);
    data(1, 3) = helpers::math::ratio(
        filter.data(0, 2) + filter.data(1, 2) + filter.data(2, 2), 2);

    data(2, 0) = helpers::math::ratio(
        filter.data(0, 0) - filter.data(1, 0) + filter.data(2, 0), 2);
    data(2, 1) = helpers::math::ratio(
        filter.data(0, 0) + filter.data(0, 1) + filter.data(0, 2) -
            filter.data(1, 0) - filter.data(1, 1) - filter.data(1, 2) +
            filter.data(2, 0) + filter.data(2, 1) + filter.data(2, 2),
        4);
    data(2, 2) = helpers::math::ratio(
        filter.data(0, 0) - filter.data(0, 1) + filter.data(0, 2) -
            filter.data(1, 0) + filter.data(1, 1) - filter.data(1, 2) +
            filter.data(2, 0) - filter.data(2, 1) + filter.data(2, 2),
        4);
    data(2, 3) = helpers::math::ratio(
        filter.data(0, 2) - filter.data(1, 2) + filter.data(2, 2), 2);

    data(3, 0) = filter.data(2, 0);
    data(3, 1) = helpers::math::ratio(
        filter.data(2, 0) + filter.data(2, 1) + filter.data(2, 2), 2);
    data(3, 2) = helpers::math::ratio(
        filter.data(2, 0) - filter.data(2, 1) + filter.data(2, 2), 2);
    data(3, 3) = filter.data(2, 2);
  }
};

template <typename T>
struct TransformedInputTile<T, 2, 2, 3, 3> final
    : public BaseTransformedInputTile<T, 2, 2, 3, 3> {
  using BaseTransformedInputTile<T, 2, 2, 3, 3>::data;
  /**
   * Apply the Winograd transform to the filter tile.
   */
  SNN_ALWAYS_INLINE explicit TransformedInputTile(
      InputTile<T, 2, 2, 3, 3> const& inp)
      : BaseTransformedInputTile<T, 2, 2, 3, 3>{} {
    data(0, 0) =
        inp.data(0, 0) + inp.data(2, 2) - inp.data(0, 2) - inp.data(2, 0);
    data(0, 1) =
        inp.data(0, 1) + inp.data(0, 2) - inp.data(2, 1) - inp.data(2, 2);
    data(0, 2) =
        inp.data(0, 2) + inp.data(2, 1) - inp.data(0, 1) - inp.data(2, 2);
    data(0, 3) =
        inp.data(0, 3) + inp.data(2, 1) - inp.data(0, 1) - inp.data(2, 3);

    data(1, 0) =
        inp.data(1, 0) + inp.data(2, 0) - inp.data(1, 2) - inp.data(2, 2);
    data(1, 1) =
        inp.data(1, 1) + inp.data(1, 2) + inp.data(2, 1) + inp.data(2, 2);
    data(1, 2) =
        inp.data(1, 2) + inp.data(2, 2) - inp.data(1, 1) - inp.data(2, 1);
    data(1, 3) =
        inp.data(1, 3) + inp.data(2, 3) - inp.data(1, 1) - inp.data(2, 1);

    data(2, 0) =
        inp.data(1, 2) + inp.data(2, 0) - inp.data(1, 0) - inp.data(2, 2);
    data(2, 1) =
        inp.data(2, 1) + inp.data(2, 2) - inp.data(1, 1) - inp.data(1, 2);
    data(2, 2) =
        inp.data(1, 1) + inp.data(2, 2) - inp.data(1, 2) - inp.data(2, 1);
    data(2, 3) =
        inp.data(1, 1) + inp.data(2, 3) - inp.data(1, 3) - inp.data(2, 1);

    data(3, 0) =
        inp.data(1, 2) + inp.data(3, 0) - inp.data(1, 0) - inp.data(3, 2);
    data(3, 1) =
        inp.data(3, 1) + inp.data(3, 2) - inp.data(1, 1) - inp.data(1, 2);
    data(3, 2) =
        inp.data(1, 1) + inp.data(3, 2) - inp.data(1, 2) - inp.data(3, 1);
    data(3, 3) =
        inp.data(1, 1) + inp.data(3, 3) - inp.data(1, 3) - inp.data(3, 1);
  }
};

template <typename T>
struct OutputTile<T, 2, 2, 3, 3> final : public BaseOutputTile<T, 2, 2, 3, 3> {
  using BaseOutputTile<T, 2, 2, 3, 3>::data;
  /**
   * Apply the Winograd transform to the intermediate tile to give the final
   * output tile.
   */
  SNN_ALWAYS_INLINE explicit OutputTile(
      IntermediateTile<T, 2, 2, 3, 3> const& tile)
      : BaseOutputTile<T, 2, 2, 3, 3>{} {
    data(0, 0) = tile.data(0, 0) + tile.data(0, 1) + tile.data(0, 2) +
                 tile.data(1, 0) + tile.data(1, 1) + tile.data(1, 2) +
                 tile.data(2, 0) + tile.data(2, 1) + tile.data(2, 2);
    data(0, 1) = tile.data(0, 1) - tile.data(0, 2) + tile.data(0, 3) +
                 tile.data(1, 1) - tile.data(1, 2) + tile.data(1, 3) +
                 tile.data(2, 1) - tile.data(2, 2) + tile.data(2, 3);
    data(1, 0) = tile.data(1, 0) + tile.data(1, 1) + tile.data(1, 2) -
                 tile.data(2, 0) - tile.data(2, 1) - tile.data(2, 2) +
                 tile.data(3, 0) + tile.data(3, 1) + tile.data(3, 2);
    data(1, 1) = tile.data(1, 1) - tile.data(1, 2) + tile.data(1, 3) -
                 tile.data(2, 1) + tile.data(2, 2) - tile.data(2, 3) +
                 tile.data(3, 1) - tile.data(3, 2) + tile.data(3, 3);
  }
};

template <typename T>
struct TransformedFilterTile<T, 2, 1, 3, 1> final
    : public BaseTransformedFilterTile<T, 2, 1, 3, 1> {
  using BaseTransformedFilterTile<T, 2, 1, 3, 1>::data;

  template <typename ConvType>
  SNN_ALWAYS_INLINE explicit TransformedFilterTile(
      FilterTile<T, 2, 1, 3, 1, ConvType> const& filter)
      : BaseTransformedFilterTile<T, 2, 1, 3, 1>{} {
    data(0, 0) = filter.data(0, 0);
    data(1, 0) = helpers::math::ratio(
        filter.data(0, 0) + filter.data(1, 0) + filter.data(2, 0), 2);
    data(2, 0) = helpers::math::ratio(
        filter.data(0, 0) - filter.data(1, 0) + filter.data(2, 0), 2);
    data(3, 0) = filter.data(2, 0);
  }
};

template <typename T>
struct TransformedInputTile<T, 2, 1, 3, 1> final
    : public BaseTransformedInputTile<T, 2, 1, 3, 1> {
  using BaseTransformedInputTile<T, 2, 1, 3, 1>::data;

  SNN_ALWAYS_INLINE explicit TransformedInputTile(
      InputTile<T, 2, 1, 3, 1> const& inp)
      : BaseTransformedInputTile<T, 2, 1, 3, 1>{} {
    data(0, 0) = inp.data(0, 0) - inp.data(2, 0);
    data(1, 0) = inp.data(1, 0) + inp.data(2, 0);
    data(2, 0) = inp.data(2, 0) - inp.data(1, 0);
    data(3, 0) = inp.data(3, 0) - inp.data(1, 0);
  }
};

template <typename T>
struct OutputTile<T, 2, 1, 3, 1> final : public BaseOutputTile<T, 2, 1, 3, 1> {
  using BaseOutputTile<T, 2, 1, 3, 1>::data;
  SNN_ALWAYS_INLINE explicit OutputTile(
      IntermediateTile<T, 2, 1, 3, 1> const& tile)
      : BaseOutputTile<T, 2, 1, 3, 1>{} {
    data(0, 0) = tile.data(0, 0) + tile.data(1, 0) + tile.data(2, 0);
    data(1, 0) = tile.data(1, 0) - tile.data(2, 0) + tile.data(3, 0);
  }
};

template <typename T>
struct TransformedFilterTile<T, 1, 2, 1, 3> final
    : public BaseTransformedFilterTile<T, 1, 2, 1, 3> {
  using BaseTransformedFilterTile<T, 1, 2, 1, 3>::data;

  template <typename ConvType>
  SNN_ALWAYS_INLINE explicit TransformedFilterTile(
      FilterTile<T, 1, 2, 1, 3, ConvType> const& filter)
      : BaseTransformedFilterTile<T, 1, 2, 1, 3>{} {
    data(0, 0) = filter.data(0, 0);
    data(0, 1) = helpers::math::ratio(
        filter.data(0, 0) + filter.data(0, 1) + filter.data(0, 2), 2);
    data(0, 2) = helpers::math::ratio(
        filter.data(0, 0) - filter.data(0, 1) + filter.data(0, 2), 2);
    data(0, 3) = filter.data(0, 2);
  }
};

template <typename T>
struct TransformedInputTile<T, 1, 2, 1, 3> final
    : public BaseTransformedInputTile<T, 1, 2, 1, 3> {
  using BaseTransformedInputTile<T, 1, 2, 1, 3>::data;

  SNN_ALWAYS_INLINE explicit TransformedInputTile(
      InputTile<T, 1, 2, 1, 3> const& inp)
      : BaseTransformedInputTile<T, 1, 2, 1, 3>{} {
    data(0, 0) = inp.data(0, 0) - inp.data(0, 2);
    data(0, 1) = inp.data(0, 1) + inp.data(0, 2);
    data(0, 2) = inp.data(0, 2) - inp.data(0, 1);
    data(0, 3) = inp.data(0, 3) - inp.data(0, 1);
  }
};

template <typename T>
struct OutputTile<T, 1, 2, 1, 3> final : public BaseOutputTile<T, 1, 2, 1, 3> {
  using BaseOutputTile<T, 1, 2, 1, 3>::data;
  SNN_ALWAYS_INLINE explicit OutputTile(
      IntermediateTile<T, 1, 2, 1, 3> const& tile)
      : BaseOutputTile<T, 1, 2, 1, 3>{} {
    data(0, 0) = tile.data(0, 0) + tile.data(0, 1) + tile.data(0, 2);
    data(0, 1) = tile.data(0, 1) - tile.data(0, 2) + tile.data(0, 3);
  }
};

template <typename T>
struct TransformedFilterTile<T, 3, 1, 2, 1> final
    : public BaseTransformedFilterTile<T, 3, 1, 2, 1> {
  using BaseTransformedFilterTile<T, 3, 1, 2, 1>::data;

  template <typename ConvType>
  SNN_ALWAYS_INLINE explicit TransformedFilterTile(
      FilterTile<T, 3, 1, 2, 1, ConvType> const& filter)
      : BaseTransformedFilterTile<T, 3, 1, 2, 1>{} {
    data(0, 0) = filter.data(0, 0);
    data(1, 0) = helpers::math::ratio(filter.data(0, 0) + filter.data(1, 0), 2);
    data(2, 0) = helpers::math::ratio(filter.data(0, 0) - filter.data(1, 0), 2);
    data(3, 0) = filter.data(1, 0);
  }
};

template <typename T>
struct TransformedInputTile<T, 3, 1, 2, 1> final
    : public BaseTransformedInputTile<T, 3, 1, 2, 1> {
  using BaseTransformedInputTile<T, 3, 1, 2, 1>::data;

  SNN_ALWAYS_INLINE explicit TransformedInputTile(
      InputTile<T, 3, 1, 2, 1> const& inp)
      : BaseTransformedInputTile<T, 3, 1, 2, 1>{} {
    data(0, 0) = inp.data(0, 0) - inp.data(2, 0);
    data(1, 0) = inp.data(1, 0) + inp.data(2, 0);
    data(2, 0) = inp.data(2, 0) - inp.data(1, 0);
    data(3, 0) = inp.data(3, 0) - inp.data(1, 0);
  }
};

template <typename T>
struct OutputTile<T, 3, 1, 2, 1> final : public BaseOutputTile<T, 3, 1, 2, 1> {
  using BaseOutputTile<T, 3, 1, 2, 1>::data;
  SNN_ALWAYS_INLINE explicit OutputTile(
      IntermediateTile<T, 3, 1, 2, 1> const& tile)
      : BaseOutputTile<T, 3, 1, 2, 1>{} {
    data(0, 0) = tile.data(0, 0) + tile.data(1, 0) + tile.data(2, 0);
    data(1, 0) = tile.data(1, 0) - tile.data(2, 0);
    data(2, 0) = tile.data(1, 0) + tile.data(2, 0) + tile.data(3, 0);
  }
};

template <typename T>
struct TransformedFilterTile<T, 1, 3, 1, 2> final
    : public BaseTransformedFilterTile<T, 1, 3, 1, 2> {
  using BaseTransformedFilterTile<T, 1, 3, 1, 2>::data;

  template <typename ConvType>
  SNN_ALWAYS_INLINE explicit TransformedFilterTile(
      FilterTile<T, 1, 3, 1, 2, ConvType> const& filter)
      : BaseTransformedFilterTile<T, 1, 3, 1, 2>{} {
    data(0, 0) = filter.data(0, 0);
    data(0, 1) = helpers::math::ratio(filter.data(0, 0) + filter.data(0, 1), 2);
    data(0, 2) = helpers::math::ratio(filter.data(0, 0) - filter.data(0, 1), 2);
    data(0, 3) = filter.data(0, 1);
  }
};

template <typename T>
struct TransformedInputTile<T, 1, 3, 1, 2> final
    : public BaseTransformedInputTile<T, 1, 3, 1, 2> {
  using BaseTransformedInputTile<T, 1, 3, 1, 2>::data;

  SNN_ALWAYS_INLINE explicit TransformedInputTile(
      InputTile<T, 1, 3, 1, 2> const& inp)
      : BaseTransformedInputTile<T, 1, 3, 1, 2>{} {
    data(0, 0) = inp.data(0, 0) - inp.data(0, 2);
    data(0, 1) = inp.data(0, 1) + inp.data(0, 2);
    data(0, 2) = inp.data(0, 2) - inp.data(0, 1);
    data(0, 3) = inp.data(0, 3) - inp.data(0, 1);
  }
};

template <typename T>
struct OutputTile<T, 1, 3, 1, 2> final : public BaseOutputTile<T, 1, 3, 1, 2> {
  using BaseOutputTile<T, 1, 3, 1, 2>::data;
  SNN_ALWAYS_INLINE explicit OutputTile(
      IntermediateTile<T, 1, 3, 1, 2> const& tile)
      : BaseOutputTile<T, 1, 3, 1, 2>{} {
    data(0, 0) = tile.data(0, 0) + tile.data(0, 1) + tile.data(0, 2);
    data(0, 1) = tile.data(0, 1) - tile.data(0, 2);
    data(0, 2) = tile.data(0, 1) + tile.data(0, 2) + tile.data(0, 3);
  }
};

template <typename T>
struct TransformedFilterTile<T, 3, 3, 2, 2> final
    : public BaseTransformedFilterTile<T, 3, 3, 2, 2> {
  using BaseTransformedFilterTile<T, 3, 3, 2, 2>::data;

  template <typename ConvType>
  SNN_ALWAYS_INLINE explicit TransformedFilterTile(
      FilterTile<T, 3, 3, 2, 2, ConvType> const& filter)
      : BaseTransformedFilterTile<T, 3, 3, 2, 2>{} {
    data(0, 0) = filter.data(0, 0);
    data(0, 1) = helpers::math::ratio(filter.data(0, 0) + filter.data(0, 1), 2);
    data(0, 2) = helpers::math::ratio(filter.data(0, 0) - filter.data(0, 1), 2);
    data(0, 3) = filter.data(0, 1);

    data(1, 0) = helpers::math::ratio(filter.data(0, 0) + filter.data(1, 0), 2);
    data(1, 1) = helpers::math::ratio(filter.data(0, 0) + filter.data(0, 1) +
                                          filter.data(1, 0) + filter.data(1, 1),
                                      4);
    data(1, 2) = helpers::math::ratio(filter.data(0, 0) - filter.data(0, 1) +
                                          filter.data(1, 0) - filter.data(1, 1),
                                      4);
    data(1, 3) = helpers::math::ratio(filter.data(0, 1) + filter.data(1, 1), 2);

    data(2, 0) = helpers::math::ratio(filter.data(0, 0) - filter.data(1, 0), 2);
    data(2, 1) = helpers::math::ratio(filter.data(0, 0) + filter.data(0, 1) -
                                          filter.data(1, 0) - filter.data(1, 1),
                                      4);
    data(2, 2) = helpers::math::ratio(filter.data(0, 0) - filter.data(0, 1) -
                                          filter.data(1, 0) + filter.data(1, 1),
                                      4);
    data(2, 3) = helpers::math::ratio(filter.data(0, 1) - filter.data(1, 1), 2);

    data(3, 0) = filter.data(1, 0);
    data(3, 1) = helpers::math::ratio(filter.data(1, 0) + filter.data(1, 1), 2);
    data(3, 2) = helpers::math::ratio(filter.data(1, 0) - filter.data(1, 1), 2);
    data(3, 3) = filter.data(1, 1);
  }
};

template <typename T>
struct TransformedInputTile<T, 3, 3, 2, 2> final
    : public BaseTransformedInputTile<T, 3, 3, 2, 2> {
  using BaseTransformedInputTile<T, 3, 3, 2, 2>::data;

  SNN_ALWAYS_INLINE explicit TransformedInputTile(
      InputTile<T, 3, 3, 2, 2> const& inp)
      : BaseTransformedInputTile<T, 3, 3, 2, 2>{} {
    data(0, 0) =
        inp.data(0, 0) - inp.data(0, 2) - inp.data(2, 0) + inp.data(2, 2);
    data(0, 1) =
        inp.data(0, 1) + inp.data(0, 2) - inp.data(2, 1) - inp.data(2, 2);
    data(0, 2) =
        inp.data(0, 2) - inp.data(0, 1) + inp.data(2, 1) - inp.data(2, 2);
    data(0, 3) =
        inp.data(0, 3) - inp.data(0, 1) + inp.data(2, 1) - inp.data(2, 3);

    data(1, 0) =
        inp.data(1, 0) - inp.data(1, 2) + inp.data(2, 0) - inp.data(2, 2);
    data(1, 1) =
        inp.data(1, 1) + inp.data(1, 2) + inp.data(2, 1) + inp.data(2, 2);
    data(1, 2) =
        inp.data(1, 2) - inp.data(1, 1) - inp.data(2, 1) + inp.data(2, 2);
    data(1, 3) =
        inp.data(1, 3) - inp.data(1, 1) - inp.data(2, 1) + inp.data(2, 3);

    data(2, 0) =
        inp.data(1, 2) - inp.data(1, 0) + inp.data(2, 0) - inp.data(2, 2);
    data(2, 1) =
        inp.data(2, 1) - inp.data(1, 1) - inp.data(1, 2) + inp.data(2, 2);
    data(2, 2) =
        inp.data(1, 1) - inp.data(1, 2) - inp.data(2, 1) + inp.data(2, 2);
    data(2, 3) =
        inp.data(1, 1) - inp.data(1, 3) - inp.data(2, 1) + inp.data(2, 3);

    data(3, 0) =
        inp.data(1, 2) - inp.data(1, 0) + inp.data(3, 0) - inp.data(3, 2);
    data(3, 1) =
        inp.data(3, 1) - inp.data(1, 1) - inp.data(1, 2) + inp.data(3, 2);
    data(3, 2) =
        inp.data(1, 1) - inp.data(1, 2) - inp.data(3, 1) + inp.data(3, 2);
    data(3, 3) =
        inp.data(1, 1) - inp.data(1, 3) - inp.data(3, 1) + inp.data(3, 3);
  }
};

template <typename T>
struct OutputTile<T, 3, 3, 2, 2> final : public BaseOutputTile<T, 3, 3, 2, 2> {
  using BaseOutputTile<T, 3, 3, 2, 2>::data;
  SNN_ALWAYS_INLINE explicit OutputTile(
      IntermediateTile<T, 3, 3, 2, 2> const& tile)
      : BaseOutputTile<T, 3, 3, 2, 2>{} {
    data(0, 0) = tile.data(0, 0) + tile.data(0, 1) + tile.data(0, 2) +
                 tile.data(1, 0) + tile.data(1, 1) + tile.data(1, 2) +
                 tile.data(2, 0) + tile.data(2, 1) + tile.data(2, 2);
    data(0, 1) = tile.data(0, 1) - tile.data(0, 2) + tile.data(1, 1) -
                 tile.data(1, 2) + tile.data(2, 1) - tile.data(2, 2);
    data(0, 2) = tile.data(0, 1) + tile.data(0, 2) + tile.data(0, 3) +
                 tile.data(1, 1) + tile.data(1, 2) + tile.data(1, 3) +
                 tile.data(2, 1) + tile.data(2, 2) + tile.data(2, 3);

    data(1, 0) = tile.data(1, 0) + tile.data(1, 1) + tile.data(1, 2) -
                 tile.data(2, 0) - tile.data(2, 1) - tile.data(2, 2);
    data(1, 1) =
        tile.data(1, 1) - tile.data(1, 2) - tile.data(2, 1) + tile.data(2, 2);
    data(1, 2) = tile.data(1, 1) + tile.data(1, 2) + tile.data(1, 3) -
                 tile.data(2, 1) - tile.data(2, 2) - tile.data(2, 3);

    data(2, 0) = tile.data(1, 0) + tile.data(1, 1) + tile.data(1, 2) +
                 tile.data(2, 0) + tile.data(2, 1) + tile.data(2, 2) +
                 tile.data(3, 0) + tile.data(3, 1) + tile.data(3, 2);
    data(2, 1) = tile.data(1, 1) - tile.data(1, 2) + tile.data(2, 1) -
                 tile.data(2, 2) + tile.data(3, 1) - tile.data(3, 2);
    data(2, 2) = tile.data(1, 1) + tile.data(1, 2) + tile.data(1, 3) +
                 tile.data(2, 1) + tile.data(2, 2) + tile.data(2, 3) +
                 tile.data(3, 1) + tile.data(3, 2) + tile.data(3, 3);
  }
};

template <typename T>
struct TransformedFilterTile<T, 3, 3, 3, 3> final
    : public BaseTransformedFilterTile<T, 3, 3, 3, 3> {
  using BaseTransformedFilterTile<T, 3, 3, 3, 3>::data;

  template <typename ConvType>
  SNN_ALWAYS_INLINE explicit TransformedFilterTile(
      FilterTile<T, 3, 3, 3, 3, ConvType> const& fil)
      : BaseTransformedFilterTile<T, 3, 3, 3, 3>{} {
    data(0, 0) = fil.data(0, 0) / 4;
    data(0, 1) = -(fil.data(0, 0) + fil.data(0, 1) + fil.data(0, 2)) / 4;
    data(0, 2) = helpers::math::ratio(
        fil.data(0, 1) - fil.data(0, 0) - fil.data(0, 2), 12);
    data(0, 3) = helpers::math::ratio(
        fil.data(0, 0) + fil.data(0, 1) * 2 + fil.data(0, 2) * 4, 12);
    data(0, 4) = fil.data(0, 2) / 2;

    data(1, 0) = -(fil.data(0, 0) + fil.data(1, 0) + fil.data(2, 0)) / 4;
    data(1, 1) = (fil.data(0, 0) + fil.data(0, 1) + fil.data(0, 2) +
                  fil.data(1, 0) + fil.data(1, 1) + fil.data(1, 2) +
                  fil.data(2, 0) + fil.data(2, 1) + fil.data(2, 2)) /
                 4;
    data(1, 2) = helpers::math::ratio(
        fil.data(0, 0) - fil.data(0, 1) + fil.data(0, 2) + fil.data(1, 0) -
            fil.data(1, 1) + fil.data(1, 2) + fil.data(2, 0) - fil.data(2, 1) +
            fil.data(2, 2),
        12);
    data(1, 3) = helpers::math::ratio(
        -fil.data(0, 0) - fil.data(1, 0) - fil.data(2, 0) -
            (fil.data(0, 1) + fil.data(1, 1) + fil.data(2, 1)) * 2 -
            (fil.data(0, 2) + fil.data(1, 2) + fil.data(2, 2)) * 4,
        12);
    data(1, 4) = -(fil.data(0, 2) + fil.data(1, 2) + fil.data(2, 2)) / 2;

    data(2, 0) = helpers::math::ratio(
        fil.data(1, 0) - fil.data(0, 0) - fil.data(2, 0), 12);
    data(2, 1) = helpers::math::ratio(
        fil.data(0, 0) + fil.data(0, 1) + fil.data(0, 2) - fil.data(1, 0) -
            fil.data(1, 1) - fil.data(1, 2) + fil.data(2, 0) + fil.data(2, 1) +
            fil.data(2, 2),
        12);
    data(2, 2) = helpers::math::ratio(
        fil.data(0, 0) - fil.data(0, 1) + fil.data(0, 2) - fil.data(1, 0) +
            fil.data(1, 1) - fil.data(1, 2) + fil.data(2, 0) - fil.data(2, 1) +
            fil.data(2, 2),
        36);
    data(2, 3) = helpers::math::ratio(
        fil.data(1, 0) - fil.data(0, 0) - fil.data(2, 0) +
            (fil.data(1, 1) - fil.data(0, 1) - fil.data(2, 1)) * 2 +
            (fil.data(1, 2) - fil.data(0, 2) - fil.data(2, 2)) * 4,
        36);
    data(2, 4) = helpers::math::ratio(
        fil.data(1, 2) - fil.data(0, 2) - fil.data(2, 2), 6);

    data(3, 0) = helpers::math::ratio(
        fil.data(0, 0) + fil.data(1, 0) * 2 + fil.data(2, 0) * 4, 12);
    data(3, 1) = -helpers::math::ratio(
        fil.data(0, 0) + fil.data(0, 1) + fil.data(0, 2) +
            (fil.data(1, 0) + fil.data(1, 1) + fil.data(1, 2)) * 2 +
            (fil.data(2, 0) + fil.data(2, 1) + fil.data(2, 2)) * 4,
        12);
    data(3, 2) = helpers::math::ratio(
        fil.data(0, 1) - fil.data(0, 0) - fil.data(0, 2) +
            (fil.data(1, 1) - fil.data(1, 0) - fil.data(1, 2)) * 2 +
            (fil.data(2, 1) - fil.data(2, 0) - fil.data(2, 2)) * 4,
        36);
    data(3, 3) = helpers::math::ratio(
        fil.data(0, 0) + 2 * (fil.data(0, 1) + fil.data(1, 0)) +
            (fil.data(0, 2) + fil.data(1, 1) + fil.data(2, 0)) * 4 +
            (fil.data(1, 2) + fil.data(2, 1)) * 8 + fil.data(2, 2) * 16,
        36);
    data(3, 4) = helpers::math::ratio(
        fil.data(0, 2) + fil.data(1, 2) * 2 + fil.data(2, 2) * 4, 6);

    data(4, 0) = fil.data(2, 0) / 2;
    data(4, 1) = -(fil.data(2, 0) + fil.data(2, 1) + fil.data(2, 2)) / 2;
    data(4, 2) = helpers::math::ratio(
        fil.data(2, 1) - fil.data(2, 0) - fil.data(2, 2), 6);
    data(4, 3) = helpers::math::ratio(
        fil.data(2, 0) + fil.data(2, 1) * 2 + fil.data(2, 2) * 4, 6);
    data(4, 4) = fil.data(2, 2);
  }
};

template <typename T>
struct TransformedInputTile<T, 3, 3, 3, 3> final
    : public BaseTransformedInputTile<T, 3, 3, 3, 3> {
  using BaseTransformedInputTile<T, 3, 3, 3, 3>::data;

  SNN_ALWAYS_INLINE explicit TransformedInputTile(
      InputTile<T, 3, 3, 3, 3> const& inp)
      : BaseTransformedInputTile<T, 3, 3, 3, 3>{} {
    data(0, 0) =
        inp.data(1, 1) - inp.data(1, 3) - inp.data(3, 1) + inp.data(3, 3) +
        (inp.data(0, 3) - inp.data(0, 1) - inp.data(1, 0) + inp.data(1, 2) +
         inp.data(2, 1) - inp.data(2, 3) + inp.data(3, 0) - inp.data(3, 2)) *
            2 +
        (inp.data(0, 0) - inp.data(0, 2) - inp.data(2, 0) + inp.data(2, 2)) * 4;

    data(0, 1) = inp.data(1, 2) - inp.data(1, 3) - inp.data(3, 2) +
                 inp.data(3, 3) +
                 (inp.data(0, 3) - inp.data(0, 2) + inp.data(1, 1) +
                  inp.data(2, 2) - inp.data(2, 3) - inp.data(3, 1)) *
                     2 +
                 (inp.data(2, 1) - inp.data(0, 1)) * 4;
    data(0, 2) =
        inp.data(3, 3) - inp.data(1, 3) +
        (inp.data(0, 3) - inp.data(1, 1) - inp.data(2, 3) + inp.data(3, 1)) *
            2 +
        (inp.data(1, 2) - inp.data(3, 2)) * 3 +
        (inp.data(0, 1) - inp.data(2, 1)) * 4 +
        (inp.data(2, 2) - inp.data(0, 2)) * 6;
    data(0, 3) =
        inp.data(1, 1) - inp.data(1, 3) - inp.data(3, 1) + inp.data(3, 3) +
        (inp.data(0, 3) - inp.data(0, 1) + inp.data(2, 1) - inp.data(2, 3)) * 2;
    data(0, 4) =
        inp.data(1, 2) - inp.data(1, 4) - inp.data(3, 2) + inp.data(3, 4) +
        (inp.data(0, 4) - inp.data(0, 2) - inp.data(1, 1) + inp.data(1, 3) +
         inp.data(2, 2) - inp.data(2, 4) + inp.data(3, 1) - inp.data(3, 3)) *
            2 +
        (inp.data(0, 1) - inp.data(0, 3) - inp.data(2, 1) + inp.data(2, 3)) * 4;

    data(1, 0) = inp.data(2, 1) - inp.data(2, 3) - inp.data(3, 1) +
                 inp.data(3, 3) +
                 (inp.data(1, 1) - inp.data(1, 3) - inp.data(2, 0) +
                  inp.data(2, 2) + inp.data(3, 0) - inp.data(3, 2)) *
                     2 +
                 (inp.data(1, 2) - inp.data(1, 0)) * 4;
    data(1, 1) =
        inp.data(2, 2) - inp.data(2, 3) - inp.data(3, 2) + inp.data(3, 3) +
        (inp.data(1, 2) - inp.data(1, 3) + inp.data(2, 1) - inp.data(3, 1)) *
            2 +
        inp.data(1, 1) * 4;
    data(1, 2) = inp.data(3, 3) - inp.data(2, 3) +
                 (inp.data(3, 1) - inp.data(2, 1) - inp.data(1, 3)) * 2 +
                 (inp.data(2, 2) - inp.data(3, 2)) * 3 - inp.data(1, 1) * 4 +
                 inp.data(1, 2) * 6;

    data(1, 3) = inp.data(2, 1) - inp.data(2, 3) - inp.data(3, 1) +
                 inp.data(3, 3) + (inp.data(1, 1) - inp.data(1, 3)) * 2;

    data(1, 4) = inp.data(2, 2) - inp.data(2, 4) - inp.data(3, 2) +
                 inp.data(3, 4) +
                 (inp.data(1, 2) - inp.data(1, 4) - inp.data(2, 1) +
                  inp.data(2, 3) + inp.data(3, 1) - inp.data(3, 3)) *
                     2 +
                 (inp.data(1, 3) - inp.data(1, 1)) * 4;

    data(2, 0) =
        inp.data(3, 3) - inp.data(3, 1) +
        (inp.data(1, 3) - inp.data(1, 1) + inp.data(3, 0) - inp.data(3, 2)) *
            2 +
        (inp.data(2, 1) - inp.data(2, 3)) * 3 +
        (inp.data(1, 0) - inp.data(1, 2)) * 4 +
        (inp.data(2, 2) - inp.data(2, 0)) * 6;
    data(2, 1) = inp.data(3, 3) - inp.data(3, 2) +
                 (inp.data(1, 3) - inp.data(1, 2) - inp.data(3, 1)) * 2 +
                 (inp.data(2, 2) - inp.data(2, 3)) * 3 - inp.data(1, 1) * 4 +
                 inp.data(2, 1) * 6;
    data(2, 2) = inp.data(3, 3) + (inp.data(1, 3) + inp.data(3, 1)) * 2 -
                 (inp.data(2, 3) + inp.data(3, 2)) * 3 + inp.data(1, 1) * 4 -
                 (inp.data(1, 2) + inp.data(2, 1)) * 6 + inp.data(2, 2) * 9;
    data(2, 3) = inp.data(3, 3) - inp.data(3, 1) +
                 (inp.data(1, 3) - inp.data(1, 1)) * 2 +
                 (inp.data(2, 1) - inp.data(2, 3)) * 3;
    data(2, 4) =
        inp.data(3, 4) - inp.data(3, 2) +
        (inp.data(1, 4) - inp.data(1, 2) + inp.data(3, 1) - inp.data(3, 3)) *
            2 +
        (inp.data(2, 2) - inp.data(2, 4)) * 3 +
        (inp.data(1, 1) - inp.data(1, 3)) * 4 +
        (inp.data(2, 3) - inp.data(2, 1)) * 6;

    data(3, 0) =
        inp.data(1, 1) - inp.data(1, 3) - inp.data(3, 1) + inp.data(3, 3) +
        (inp.data(1, 2) - inp.data(1, 0) + inp.data(3, 0) - inp.data(3, 2)) * 2;
    data(3, 1) = inp.data(1, 2) - inp.data(1, 3) - inp.data(3, 2) +
                 inp.data(3, 3) + (inp.data(1, 1) - inp.data(3, 1)) * 2;
    data(3, 2) = inp.data(3, 3) - inp.data(1, 3) +
                 (inp.data(3, 1) - inp.data(1, 1)) * 2 +
                 (inp.data(1, 2) - inp.data(3, 2)) * 3;
    data(3, 3) =
        inp.data(1, 1) - inp.data(1, 3) - inp.data(3, 1) + inp.data(3, 3);
    data(3, 4) =
        inp.data(1, 2) - inp.data(1, 4) - inp.data(3, 2) + inp.data(3, 4) +
        (inp.data(1, 3) - inp.data(1, 1) + inp.data(3, 1) - inp.data(3, 3)) * 2;

    data(4, 0) =
        inp.data(2, 1) - inp.data(2, 3) - inp.data(4, 1) + inp.data(4, 3) +
        (inp.data(1, 3) - inp.data(1, 1) - inp.data(2, 0) + inp.data(2, 2) +
         inp.data(3, 1) - inp.data(3, 3) + inp.data(4, 0) - inp.data(4, 2)) *
            2 +
        (inp.data(1, 0) - inp.data(1, 2) - inp.data(3, 0) + inp.data(3, 2)) * 4;
    data(4, 1) = inp.data(2, 2) - inp.data(2, 3) - inp.data(4, 2) +
                 inp.data(4, 3) +
                 (inp.data(1, 3) - inp.data(1, 2) + inp.data(2, 1) +
                  inp.data(3, 2) - inp.data(3, 3) - inp.data(4, 1)) *
                     2 +
                 (inp.data(3, 1) - inp.data(1, 1)) * 4;
    data(4, 2) =
        inp.data(4, 3) - inp.data(2, 3) +
        (inp.data(1, 3) - inp.data(2, 1) - inp.data(3, 3) + inp.data(4, 1)) *
            2 +
        (inp.data(2, 2) - inp.data(4, 2)) * 3 +
        (inp.data(1, 1) - inp.data(3, 1)) * 4 +
        (inp.data(3, 2) - inp.data(1, 2)) * 6;
    data(4, 3) =
        inp.data(2, 1) - inp.data(2, 3) - inp.data(4, 1) + inp.data(4, 3) +
        (inp.data(1, 3) - inp.data(1, 1) + inp.data(3, 1) - inp.data(3, 3)) * 2;
    data(4, 4) =
        inp.data(2, 2) - inp.data(2, 4) + inp.data(4, 4) - inp.data(4, 2) +
        (inp.data(1, 4) - inp.data(1, 2) - inp.data(2, 1) + inp.data(2, 3) +
         inp.data(3, 2) - inp.data(3, 4) + inp.data(4, 1) - inp.data(4, 3)) *
            2 +
        (inp.data(1, 1) - inp.data(3, 1) - inp.data(1, 3) + inp.data(3, 3)) * 4;
  }
};

template <typename T>
struct OutputTile<T, 3, 3, 3, 3> final : public BaseOutputTile<T, 3, 3, 3, 3> {
  using BaseOutputTile<T, 3, 3, 3, 3>::data;
  SNN_ALWAYS_INLINE explicit OutputTile(
      IntermediateTile<T, 3, 3, 3, 3> const& tile)
      : BaseOutputTile<T, 3, 3, 3, 3>{} {
    data(0, 0) =
        tile.data(0, 0) + tile.data(0, 1) + tile.data(0, 2) + tile.data(0, 3) +
        tile.data(1, 0) + tile.data(1, 1) + tile.data(1, 2) + tile.data(1, 3) +
        tile.data(2, 0) + tile.data(2, 1) + tile.data(2, 2) + tile.data(2, 3) +
        tile.data(3, 0) + tile.data(3, 1) + tile.data(3, 2) + tile.data(3, 3);
    data(0, 1) = tile.data(0, 1) - tile.data(0, 2) + tile.data(1, 1) -
                 tile.data(1, 2) + tile.data(2, 1) - tile.data(2, 2) +
                 tile.data(3, 1) - tile.data(3, 2) +
                 (tile.data(0, 3) + tile.data(1, 3) + tile.data(2, 3) +
                  tile.data(3, 3)) *
                     2;
    data(0, 2) = tile.data(0, 1) + tile.data(0, 2) + tile.data(0, 4) +
                 tile.data(1, 1) + tile.data(1, 2) + tile.data(1, 4) +
                 tile.data(2, 1) + tile.data(2, 2) + tile.data(2, 4) +
                 tile.data(3, 1) + tile.data(3, 2) + tile.data(3, 4) +
                 (tile.data(0, 3) + tile.data(1, 3) + tile.data(2, 3) +
                  tile.data(3, 3)) *
                     4;

    data(1, 0) = tile.data(1, 0) + tile.data(1, 1) + tile.data(1, 2) +
                 tile.data(1, 3) - tile.data(2, 0) - tile.data(2, 1) -
                 tile.data(2, 2) - tile.data(2, 3) +
                 (tile.data(3, 0) + tile.data(3, 1) + tile.data(3, 2) +
                  tile.data(3, 3)) *
                     2;
    data(1, 1) = tile.data(1, 1) - tile.data(1, 2) - tile.data(2, 1) +
                 tile.data(2, 2) +
                 (tile.data(1, 3) - tile.data(2, 3) + tile.data(3, 1) -
                  tile.data(3, 2)) *
                     2 +
                 tile.data(3, 3) * 4;
    data(1, 2) = tile.data(1, 1) + tile.data(1, 2) + tile.data(1, 4) -
                 tile.data(2, 1) - tile.data(2, 2) - tile.data(2, 4) +
                 (tile.data(3, 1) + tile.data(3, 2) + tile.data(3, 4)) * 2 +
                 (tile.data(1, 3) - tile.data(2, 3)) * 4 + tile.data(3, 3) * 8;

    data(2, 0) = tile.data(1, 0) + tile.data(1, 1) + tile.data(1, 2) +
                 tile.data(1, 3) + tile.data(2, 0) + tile.data(2, 1) +
                 tile.data(2, 2) + tile.data(2, 3) + tile.data(4, 0) +
                 tile.data(4, 1) + tile.data(4, 2) + tile.data(4, 3) +
                 (tile.data(3, 0) + tile.data(3, 1) + tile.data(3, 2) +
                  tile.data(3, 3)) *
                     4;
    data(2, 1) = tile.data(1, 1) - tile.data(1, 2) + tile.data(2, 1) -
                 tile.data(2, 2) + tile.data(4, 1) - tile.data(4, 2) +
                 (tile.data(1, 3) + tile.data(2, 3) + tile.data(4, 3)) * 2 +
                 (tile.data(3, 1) - tile.data(3, 2)) * 4 + tile.data(3, 3) * 8;
    data(2, 2) = tile.data(1, 1) + tile.data(1, 2) + tile.data(1, 4) +
                 tile.data(2, 1) + tile.data(2, 2) + tile.data(2, 4) +
                 tile.data(4, 1) + tile.data(4, 2) + tile.data(4, 4) +
                 (tile.data(1, 3) + tile.data(2, 3) + tile.data(3, 1) +
                  tile.data(3, 2) + tile.data(3, 4) + tile.data(4, 3)) *
                     4 +
                 tile.data(3, 3) * 16;
  }
};

template <typename T>
struct TransformedFilterTile<T, 4, 4, 3, 3> final
    : public BaseTransformedFilterTile<T, 4, 4, 3, 3> {
  using BaseTransformedFilterTile<T, 4, 4, 3, 3>::data;

  template <typename ConvType>
  SNN_ALWAYS_INLINE explicit TransformedFilterTile(
      FilterTile<T, 4, 4, 3, 3, ConvType> const& filter)
      : BaseTransformedFilterTile<T, 4, 4, 3, 3>{} {
    data(0, 0) = filter.data(0, 0) / 16;
    data(0, 1) = -helpers::math::ratio(
        filter.data(0, 0) + filter.data(0, 1) + filter.data(0, 2), 24);
    data(0, 2) = -helpers::math::ratio(
        filter.data(0, 0) - filter.data(0, 1) + filter.data(0, 2), 24);
    data(0, 3) = helpers::math::ratio(
        filter.data(0, 0) + filter.data(0, 1) * 2 + filter.data(0, 2) * 4, 96);
    data(0, 4) = helpers::math::ratio(
        filter.data(0, 0) - filter.data(0, 1) * 2 + filter.data(0, 2) * 4, 96);
    data(0, 5) = filter.data(0, 2) / 4;
    data(1, 0) = -helpers::math::ratio(
        filter.data(0, 0) + filter.data(1, 0) + filter.data(2, 0), 24);
    data(1, 1) = helpers::math::ratio(
        filter.data(0, 0) + filter.data(0, 1) + filter.data(0, 2) +
            filter.data(1, 0) + filter.data(1, 1) + filter.data(1, 2) +
            filter.data(2, 0) + filter.data(2, 1) + filter.data(2, 2),
        36);
    data(1, 2) = helpers::math::ratio(
        filter.data(0, 0) - filter.data(0, 1) + filter.data(0, 2) +
            filter.data(1, 0) - filter.data(1, 1) + filter.data(1, 2) +
            filter.data(2, 0) - filter.data(2, 1) + filter.data(2, 2),
        36);
    data(1, 3) = -helpers::math::ratio(
        filter.data(0, 0) + filter.data(1, 0) + filter.data(2, 0) +
            (filter.data(0, 1) + filter.data(1, 1) + filter.data(2, 1)) * 2 +
            (filter.data(0, 2) + filter.data(1, 2) + filter.data(2, 2)) * 4,
        144);
    data(1, 4) = -helpers::math::ratio(
        filter.data(0, 0) + filter.data(1, 0) + filter.data(2, 0) -
            (filter.data(0, 1) + filter.data(1, 1) + filter.data(2, 1)) * 2 +
            (filter.data(0, 2) + filter.data(1, 2) + filter.data(2, 2)) * 4,
        144);
    data(1, 5) = -helpers::math::ratio(
        filter.data(0, 2) + filter.data(1, 2) + filter.data(2, 2), 6);
    data(2, 0) = -helpers::math::ratio(
        filter.data(0, 0) - filter.data(1, 0) + filter.data(2, 0), 24);
    data(2, 1) = helpers::math::ratio(
        filter.data(0, 0) + filter.data(0, 1) + filter.data(0, 2) -
            filter.data(1, 0) - filter.data(1, 1) - filter.data(1, 2) +
            filter.data(2, 0) + filter.data(2, 1) + filter.data(2, 2),
        36);
    data(2, 2) = helpers::math::ratio(
        filter.data(0, 0) - filter.data(0, 1) + filter.data(0, 2) -
            filter.data(1, 0) + filter.data(1, 1) - filter.data(1, 2) +
            filter.data(2, 0) - filter.data(2, 1) + filter.data(2, 2),
        36);
    data(2, 3) = -helpers::math::ratio(
        filter.data(0, 0) - filter.data(1, 0) + filter.data(2, 0) +
            (filter.data(0, 1) - filter.data(1, 1) + filter.data(2, 1)) * 2 +
            (filter.data(0, 2) - filter.data(1, 2) + filter.data(2, 2)) * 4,
        144);
    data(2, 4) = -helpers::math::ratio(
        filter.data(0, 0) - filter.data(1, 0) + filter.data(2, 0) +
            (filter.data(1, 1) - filter.data(0, 1) - filter.data(2, 1)) * 2 +
            (filter.data(0, 2) - filter.data(1, 2) + filter.data(2, 2)) * 4,
        144);
    data(2, 5) = helpers::math::ratio(
        filter.data(1, 2) - filter.data(0, 2) - filter.data(2, 2), 6);
    data(3, 0) = helpers::math::ratio(
        filter.data(0, 0) + filter.data(1, 0) * 2 + filter.data(2, 0) * 4, 96);
    data(3, 1) = -helpers::math::ratio(
        filter.data(0, 0) + filter.data(0, 1) + filter.data(0, 2) +
            (filter.data(1, 0) + filter.data(1, 1) + filter.data(1, 2)) * 2 +
            (filter.data(2, 0) + filter.data(2, 1) + filter.data(2, 2)) * 4,
        144);
    data(3, 2) = -helpers::math::ratio(
        filter.data(0, 0) - filter.data(0, 1) + filter.data(0, 2) +
            (filter.data(1, 0) - filter.data(1, 1) + filter.data(1, 2)) * 2 +
            (filter.data(2, 0) - filter.data(2, 1) + filter.data(2, 2)) * 4,
        144);
    data(3, 3) = helpers::math::ratio(
        filter.data(0, 0) + (filter.data(0, 1) + filter.data(1, 0)) * 2 +
            (filter.data(0, 2) + filter.data(1, 1) + filter.data(2, 0)) * 4 +
            (filter.data(1, 2) + filter.data(2, 1)) * 8 +
            filter.data(2, 2) * 16,
        576);
    data(3, 4) = helpers::math::ratio(
        filter.data(0, 0) + (filter.data(1, 0) - filter.data(0, 1)) * 2 +
            (filter.data(0, 2) - filter.data(1, 1) + filter.data(2, 0)) * 4 +
            (filter.data(1, 2) - filter.data(2, 1)) * 8 +
            filter.data(2, 2) * 16,
        576);
    data(3, 5) = helpers::math::ratio(
        filter.data(0, 2) + filter.data(1, 2) * 2 + filter.data(2, 2) * 4, 24);
    data(4, 0) = helpers::math::ratio(
        filter.data(0, 0) - filter.data(1, 0) * 2 + filter.data(2, 0) * 4, 96);
    data(4, 1) = -helpers::math::ratio(
        filter.data(0, 0) + filter.data(0, 1) + filter.data(0, 2) -
            (filter.data(1, 0) + filter.data(1, 1) + filter.data(1, 2)) * 2 +
            (filter.data(2, 0) + filter.data(2, 1) + filter.data(2, 2)) * 4,
        144);
    data(4, 2) = -helpers::math::ratio(
        filter.data(0, 0) - filter.data(0, 1) + filter.data(0, 2) +
            (filter.data(1, 1) - filter.data(1, 0) - filter.data(1, 2)) * 2 +
            (filter.data(2, 0) - filter.data(2, 1) + filter.data(2, 2)) * 4,
        144);
    data(4, 3) = helpers::math::ratio(
        filter.data(0, 0) + (filter.data(0, 1) - filter.data(1, 0)) * 2 +
            (filter.data(0, 2) - filter.data(1, 1) + filter.data(2, 0)) * 4 +
            (filter.data(2, 1) - filter.data(1, 2)) * 8 +
            filter.data(2, 2) * 16,
        576);
    data(4, 4) = helpers::math::ratio(
        filter.data(0, 0) + (-filter.data(0, 1) - filter.data(1, 0)) * 2 +
            (filter.data(0, 2) + filter.data(1, 1) + filter.data(2, 0)) * 4 +
            (-filter.data(1, 2) - filter.data(2, 1)) * 8 +
            filter.data(2, 2) * 16,
        576);
    data(4, 5) = helpers::math::ratio(
        filter.data(0, 2) - filter.data(1, 2) * 2 + filter.data(2, 2) * 4, 24);
    data(5, 0) = filter.data(2, 0) / 4;
    data(5, 1) = -helpers::math::ratio(
        filter.data(2, 0) + filter.data(2, 1) + filter.data(2, 2), 6);
    data(5, 2) = -helpers::math::ratio(
        filter.data(2, 0) - filter.data(2, 1) + filter.data(2, 2), 6);
    data(5, 3) = helpers::math::ratio(
        filter.data(2, 0) + filter.data(2, 1) * 2 + filter.data(2, 2) * 4, 24);
    data(5, 4) = helpers::math::ratio(
        filter.data(2, 0) - filter.data(2, 1) * 2 + filter.data(2, 2) * 4, 24);
    data(5, 5) = filter.data(2, 2);
  }
};

template <typename T>
struct TransformedInputTile<T, 4, 4, 3, 3> final
    : public BaseTransformedInputTile<T, 4, 4, 3, 3> {
  using BaseTransformedInputTile<T, 4, 4, 3, 3>::data;

  SNN_ALWAYS_INLINE explicit TransformedInputTile(
      InputTile<T, 4, 4, 3, 3> const& input)
      : BaseTransformedInputTile<T, 4, 4, 3, 3>{} {
    data(0, 0) =
        input.data(4, 4) + (input.data(0, 4) + input.data(4, 0)) * 4 -
        (input.data(2, 4) + input.data(4, 2)) * 5 + input.data(0, 0) * 16 -
        (input.data(0, 2) + input.data(2, 0)) * 20 + input.data(2, 2) * 25;
    data(0, 1) = input.data(4, 3) + input.data(4, 4) +
                 (input.data(0, 3) + input.data(0, 4) - input.data(4, 1) -
                  input.data(4, 2)) *
                     4 -
                 (input.data(2, 3) + input.data(2, 4)) * 5 -
                 (input.data(0, 1) + input.data(0, 2)) * 16 +
                 (input.data(2, 1) + input.data(2, 2)) * 20;
    data(0, 2) = input.data(4, 4) - input.data(4, 3) +
                 (input.data(0, 4) - input.data(0, 3) + input.data(4, 1) -
                  input.data(4, 2)) *
                     4 +
                 (input.data(2, 3) - input.data(2, 4)) * 5 +
                 (input.data(0, 1) - input.data(0, 2)) * 16 +
                 (input.data(2, 2) - input.data(2, 1)) * 20;
    data(0, 3) = input.data(4, 4) - input.data(4, 2) +
                 (input.data(4, 3) - input.data(4, 1)) * 2 +
                 (input.data(0, 4) - input.data(0, 2)) * 4 +
                 (input.data(2, 2) - input.data(2, 4)) * 5 +
                 (input.data(0, 3) - input.data(0, 1)) * 8 +
                 (input.data(2, 1) - input.data(2, 3)) * 10;
    data(0, 4) = input.data(4, 4) - input.data(4, 2) +
                 (input.data(4, 1) - input.data(4, 3)) * 2 +
                 (input.data(0, 4) - input.data(0, 2)) * 4 +
                 (input.data(2, 2) - input.data(2, 4)) * 5 +
                 (input.data(0, 1) - input.data(0, 3)) * 8 +
                 (input.data(2, 3) - input.data(2, 1)) * 10;
    data(0, 5) =
        input.data(4, 5) + (input.data(4, 1) + input.data(0, 5)) * 4 -
        (input.data(2, 5) + input.data(4, 3)) * 5 + input.data(0, 1) * 16 -
        (input.data(0, 3) + input.data(2, 1)) * 20 + input.data(2, 3) * 25;
    data(1, 0) = input.data(3, 4) + input.data(4, 4) +
                 (input.data(3, 0) - input.data(1, 4) - input.data(2, 4) +
                  input.data(4, 0)) *
                     4 -
                 (input.data(3, 2) + input.data(4, 2)) * 5 -
                 (input.data(1, 0) + input.data(2, 0)) * 16 +
                 (input.data(1, 2) + input.data(2, 2)) * 20;
    data(1, 1) = input.data(3, 3) + input.data(3, 4) + input.data(4, 3) +
                 input.data(4, 4) -
                 (input.data(1, 3) + input.data(1, 4) + input.data(2, 3) +
                  input.data(2, 4) + input.data(3, 1) + input.data(3, 2) +
                  input.data(4, 1) + input.data(4, 2)) *
                     4 +
                 (input.data(1, 1) + input.data(1, 2) + input.data(2, 1) +
                  input.data(2, 2)) *
                     16;
    data(1, 2) = input.data(3, 4) - input.data(3, 3) - input.data(4, 3) +
                 input.data(4, 4) +
                 (input.data(1, 3) - input.data(1, 4) + input.data(2, 3) -
                  input.data(2, 4) + input.data(3, 1) - input.data(3, 2) +
                  input.data(4, 1) - input.data(4, 2)) *
                     4 +
                 (input.data(1, 2) - input.data(1, 1) - input.data(2, 1) +
                  input.data(2, 2)) *
                     16;
    data(1, 3) = input.data(3, 4) - input.data(3, 2) - input.data(4, 2) +
                 input.data(4, 4) +
                 (input.data(3, 3) - input.data(3, 1) - input.data(4, 1) +
                  input.data(4, 3)) *
                     2 +
                 (input.data(1, 2) - input.data(1, 4) + input.data(2, 2) -
                  input.data(2, 4)) *
                     4 +
                 (input.data(1, 1) - input.data(1, 3) + input.data(2, 1) -
                  input.data(2, 3)) *
                     8;
    data(1, 4) = input.data(3, 4) - input.data(3, 2) - input.data(4, 2) +
                 input.data(4, 4) +
                 (input.data(3, 1) - input.data(3, 3) + input.data(4, 1) -
                  input.data(4, 3)) *
                     2 +
                 (input.data(1, 2) - input.data(1, 4) + input.data(2, 2) -
                  input.data(2, 4)) *
                     4 +
                 (input.data(1, 3) - input.data(1, 1) - input.data(2, 1) +
                  input.data(2, 3)) *
                     8;
    data(1, 5) = input.data(3, 5) + input.data(4, 5) +
                 (input.data(3, 1) - input.data(1, 5) - input.data(2, 5) +
                  input.data(4, 1)) *
                     4 -
                 (input.data(3, 3) + input.data(4, 3)) * 5 -
                 (input.data(1, 1) + input.data(2, 1)) * 16 +
                 (input.data(1, 3) + input.data(2, 3)) * 20;
    data(2, 0) = input.data(4, 4) - input.data(3, 4) +
                 (input.data(1, 4) - input.data(2, 4) - input.data(3, 0) +
                  input.data(4, 0)) *
                     4 +
                 (input.data(3, 2) - input.data(4, 2)) * 5 +
                 (input.data(1, 0) - input.data(2, 0)) * 16 +
                 (input.data(2, 2) - input.data(1, 2)) * 20;
    data(2, 1) = input.data(4, 3) - input.data(3, 3) - input.data(3, 4) +
                 input.data(4, 4) +
                 (input.data(1, 3) + input.data(1, 4) - input.data(2, 3) -
                  input.data(2, 4) + input.data(3, 1) + input.data(3, 2) -
                  input.data(4, 1) - input.data(4, 2)) *
                     4 +
                 (input.data(2, 1) - input.data(1, 1) - input.data(1, 2) +
                  input.data(2, 2)) *
                     16;
    data(2, 2) = input.data(3, 3) - input.data(3, 4) - input.data(4, 3) +
                 input.data(4, 4) +
                 (input.data(1, 4) - input.data(1, 3) + input.data(2, 3) -
                  input.data(2, 4) - input.data(3, 1) + input.data(3, 2) +
                  input.data(4, 1) - input.data(4, 2)) *
                     4 +
                 (input.data(1, 1) - input.data(1, 2) - input.data(2, 1) +
                  input.data(2, 2)) *
                     16;
    data(2, 3) = input.data(3, 2) - input.data(3, 4) - input.data(4, 2) +
                 input.data(4, 4) +
                 (input.data(3, 1) - input.data(3, 3) - input.data(4, 1) +
                  input.data(4, 3)) *
                     2 +
                 (input.data(1, 4) - input.data(1, 2) + input.data(2, 2) -
                  input.data(2, 4)) *
                     4 +
                 (input.data(1, 3) - input.data(1, 1) + input.data(2, 1) -
                  input.data(2, 3)) *
                     8;
    data(2, 4) = input.data(3, 2) - input.data(3, 4) - input.data(4, 2) +
                 input.data(4, 4) +
                 (input.data(3, 3) - input.data(3, 1) + input.data(4, 1) -
                  input.data(4, 3)) *
                     2 +
                 (input.data(1, 4) - input.data(1, 2) + input.data(2, 2) -
                  input.data(2, 4)) *
                     4 +
                 (input.data(1, 1) - input.data(1, 3) - input.data(2, 1) +
                  input.data(2, 3)) *
                     8;
    data(2, 5) = input.data(4, 5) - input.data(3, 5) +
                 (input.data(1, 5) - input.data(2, 5) - input.data(3, 1) +
                  input.data(4, 1)) *
                     4 +
                 (input.data(3, 3) - input.data(4, 3)) * 5 +
                 (input.data(1, 1) - input.data(2, 1)) * 16 +
                 (input.data(2, 3) - input.data(1, 3)) * 20;
    data(3, 0) = input.data(4, 4) - input.data(2, 4) +
                 (input.data(3, 4) - input.data(1, 4)) * 2 +
                 (input.data(4, 0) - input.data(2, 0)) * 4 +
                 (input.data(2, 2) - input.data(4, 2)) * 5 +
                 (input.data(3, 0) - input.data(1, 0)) * 8 +
                 (input.data(1, 2) - input.data(3, 2)) * 10;
    data(3, 1) = input.data(4, 3) - input.data(2, 3) - input.data(2, 4) +
                 input.data(4, 4) +
                 (input.data(3, 3) + input.data(3, 4) - input.data(1, 3) -
                  input.data(1, 4)) *
                     2 +
                 (input.data(2, 1) + input.data(2, 2) - input.data(4, 1) -
                  input.data(4, 2)) *
                     4 +
                 (input.data(1, 1) + input.data(1, 2) - input.data(3, 1) -
                  input.data(3, 2)) *
                     8;
    data(3, 2) = input.data(2, 3) - input.data(2, 4) - input.data(4, 3) +
                 input.data(4, 4) +
                 (input.data(1, 3) - input.data(1, 4) - input.data(3, 3) +
                  input.data(3, 4)) *
                     2 +
                 (input.data(2, 2) - input.data(2, 1) + input.data(4, 1) -
                  input.data(4, 2)) *
                     4 +
                 (input.data(1, 2) - input.data(1, 1) + input.data(3, 1) -
                  input.data(3, 2)) *
                     8;
    data(3, 3) = input.data(2, 2) - input.data(2, 4) - input.data(4, 2) +
                 input.data(4, 4) +
                 (input.data(1, 2) - input.data(1, 4) + input.data(2, 1) -
                  input.data(2, 3) - input.data(3, 2) + input.data(3, 4) -
                  input.data(4, 1) + input.data(4, 3)) *
                     2 +
                 (input.data(1, 1) - input.data(1, 3) - input.data(3, 1) +
                  input.data(3, 3)) *
                     4;
    data(3, 4) = input.data(2, 2) - input.data(2, 4) - input.data(4, 2) +
                 input.data(4, 4) +
                 (input.data(1, 2) - input.data(1, 4) - input.data(2, 1) +
                  input.data(2, 3) - input.data(3, 2) + input.data(3, 4) +
                  input.data(4, 1) - input.data(4, 3)) *
                     2 +
                 (input.data(1, 3) - input.data(1, 1) + input.data(3, 1) -
                  input.data(3, 3)) *
                     4;
    data(3, 5) = input.data(4, 5) - input.data(2, 5) +
                 (input.data(3, 5) - input.data(1, 5)) * 2 +
                 (input.data(4, 1) - input.data(2, 1)) * 4 +
                 (input.data(2, 3) - input.data(4, 3)) * 5 +
                 (input.data(3, 1) - input.data(1, 1)) * 8 +
                 (input.data(1, 3) - input.data(3, 3)) * 10;
    data(4, 0) = input.data(4, 4) - input.data(2, 4) +
                 (input.data(1, 4) - input.data(3, 4)) * 2 +
                 (input.data(4, 0) - input.data(2, 0)) * 4 +
                 (input.data(2, 2) - input.data(4, 2)) * 5 +
                 (input.data(1, 0) - input.data(3, 0)) * 8 +
                 (input.data(3, 2) - input.data(1, 2)) * 10;
    data(4, 1) = input.data(4, 3) - input.data(2, 3) - input.data(2, 4) +
                 input.data(4, 4) +
                 (input.data(1, 3) + input.data(1, 4) - input.data(3, 3) -
                  input.data(3, 4)) *
                     2 +
                 (input.data(2, 1) + input.data(2, 2) - input.data(4, 1) -
                  input.data(4, 2)) *
                     4 +
                 (input.data(3, 1) - input.data(1, 1) - input.data(1, 2) +
                  input.data(3, 2)) *
                     8;
    data(4, 2) = input.data(2, 3) - input.data(2, 4) - input.data(4, 3) +
                 input.data(4, 4) +
                 (input.data(1, 4) - input.data(1, 3) + input.data(3, 3) -
                  input.data(3, 4)) *
                     2 +
                 (input.data(2, 2) - input.data(2, 1) + input.data(4, 1) -
                  input.data(4, 2)) *
                     4 +
                 (input.data(1, 1) - input.data(1, 2) - input.data(3, 1) +
                  input.data(3, 2)) *
                     8;
    data(4, 3) = input.data(2, 2) - input.data(2, 4) - input.data(4, 2) +
                 input.data(4, 4) +
                 (input.data(1, 4) - input.data(1, 2) + input.data(2, 1) -
                  input.data(2, 3) + input.data(3, 2) - input.data(3, 4) -
                  input.data(4, 1) + input.data(4, 3)) *
                     2 +
                 (input.data(1, 3) - input.data(1, 1) + input.data(3, 1) -
                  input.data(3, 3)) *
                     4;
    data(4, 4) = input.data(2, 2) - input.data(2, 4) - input.data(4, 2) +
                 input.data(4, 4) +
                 (input.data(1, 4) - input.data(1, 2) - input.data(2, 1) +
                  input.data(2, 3) + input.data(3, 2) - input.data(3, 4) +
                  input.data(4, 1) - input.data(4, 3)) *
                     2 +
                 (input.data(1, 1) - input.data(1, 3) - input.data(3, 1) +
                  input.data(3, 3)) *
                     4;
    data(4, 5) = input.data(4, 5) - input.data(2, 5) +
                 (input.data(1, 5) - input.data(3, 5)) * 2 +
                 (input.data(4, 1) - input.data(2, 1)) * 4 +
                 (input.data(2, 3) - input.data(4, 3)) * 5 +
                 (input.data(1, 1) - input.data(3, 1)) * 8 +
                 (input.data(3, 3) - input.data(1, 3)) * 10;
    data(5, 0) =
        input.data(5, 4) + (input.data(1, 4) + input.data(5, 0)) * 4 -
        (input.data(3, 4) + input.data(5, 2)) * 5 + input.data(1, 0) * 16 -
        (input.data(1, 2) + input.data(3, 0)) * 20 + input.data(3, 2) * 25;
    data(5, 1) = input.data(5, 3) + input.data(5, 4) +
                 (input.data(1, 3) - input.data(5, 1) - input.data(5, 2) +
                  input.data(1, 4)) *
                     4 -
                 (input.data(3, 3) + input.data(3, 4)) * 5 -
                 (input.data(1, 1) + input.data(1, 2)) * 16 +
                 (input.data(3, 1) + input.data(3, 2)) * 20;
    data(5, 2) = input.data(5, 4) - input.data(5, 3) +
                 (input.data(1, 4) - input.data(1, 3) + input.data(5, 1) -
                  input.data(5, 2)) *
                     4 +
                 (input.data(3, 3) - input.data(3, 4)) * 5 +
                 (input.data(1, 1) - input.data(1, 2)) * 16 +
                 (input.data(3, 2) - input.data(3, 1)) * 20;
    data(5, 3) = input.data(5, 4) - input.data(5, 2) +
                 (input.data(5, 3) - input.data(5, 1)) * 2 +
                 (input.data(1, 4) - input.data(1, 2)) * 4 +
                 (input.data(3, 2) - input.data(3, 4)) * 5 +
                 (input.data(1, 3) - input.data(1, 1)) * 8 +
                 (input.data(3, 1) - input.data(3, 3)) * 10;
    data(5, 4) = input.data(5, 4) - input.data(5, 2) +
                 (input.data(5, 1) - input.data(5, 3)) * 2 +
                 (input.data(1, 4) - input.data(1, 2)) * 4 +
                 (input.data(3, 2) - input.data(3, 4)) * 5 +
                 (input.data(1, 1) - input.data(1, 3)) * 8 +
                 (input.data(3, 3) - input.data(3, 1)) * 10;
    data(5, 5) =
        input.data(5, 5) + (input.data(5, 1) + input.data(1, 5)) * 4 -
        (input.data(3, 5) + input.data(5, 3)) * 5 + input.data(1, 1) * 16 -
        (input.data(1, 3) + input.data(3, 1)) * 20 + input.data(3, 3) * 25;
  }
};

template <typename T>
struct OutputTile<T, 4, 4, 3, 3> final : public BaseOutputTile<T, 4, 4, 3, 3> {
  using BaseOutputTile<T, 4, 4, 3, 3>::data;
  SNN_ALWAYS_INLINE explicit OutputTile(
      IntermediateTile<T, 4, 4, 3, 3> const& inter)
      : BaseOutputTile<T, 4, 4, 3, 3>{} {
    data(0, 0) = inter.data(0, 0) + inter.data(0, 1) + inter.data(0, 2) +
                 inter.data(0, 3) + inter.data(0, 4) + inter.data(1, 0) +
                 inter.data(1, 1) + inter.data(1, 2) + inter.data(1, 3) +
                 inter.data(1, 4) + inter.data(2, 0) + inter.data(2, 1) +
                 inter.data(2, 2) + inter.data(2, 3) + inter.data(2, 4) +
                 inter.data(3, 0) + inter.data(3, 1) + inter.data(3, 2) +
                 inter.data(3, 3) + inter.data(3, 4) + inter.data(4, 0) +
                 inter.data(4, 1) + inter.data(4, 2) + inter.data(4, 3) +
                 inter.data(4, 4);
    data(0, 1) = inter.data(0, 1) - inter.data(0, 2) + inter.data(1, 1) -
                 inter.data(1, 2) + inter.data(2, 1) - inter.data(2, 2) +
                 inter.data(3, 1) - inter.data(3, 2) + inter.data(4, 1) -
                 inter.data(4, 2) +
                 (inter.data(0, 3) - inter.data(0, 4) + inter.data(1, 3) -
                  inter.data(1, 4) + inter.data(2, 3) - inter.data(2, 4) +
                  inter.data(3, 3) - inter.data(3, 4) + inter.data(4, 3) -
                  inter.data(4, 4)) *
                     2;
    data(0, 2) = inter.data(0, 1) + inter.data(0, 2) + inter.data(1, 1) +
                 inter.data(1, 2) + inter.data(2, 1) + inter.data(2, 2) +
                 inter.data(3, 1) + inter.data(3, 2) + inter.data(4, 1) +
                 inter.data(4, 2) +
                 (inter.data(0, 3) + inter.data(0, 4) + inter.data(1, 3) +
                  inter.data(1, 4) + inter.data(2, 3) + inter.data(2, 4) +
                  inter.data(3, 3) + inter.data(3, 4) + inter.data(4, 3) +
                  inter.data(4, 4)) *
                     4;
    data(0, 3) = inter.data(0, 1) - inter.data(0, 2) + inter.data(0, 5) +
                 inter.data(1, 1) - inter.data(1, 2) + inter.data(1, 5) +
                 inter.data(2, 1) - inter.data(2, 2) + inter.data(2, 5) +
                 inter.data(3, 1) - inter.data(3, 2) + inter.data(3, 5) +
                 inter.data(4, 1) - inter.data(4, 2) + inter.data(4, 5) +
                 (inter.data(0, 3) - inter.data(0, 4) + inter.data(1, 3) -
                  inter.data(1, 4) + inter.data(2, 3) - inter.data(2, 4) +
                  inter.data(3, 3) - inter.data(3, 4) + inter.data(4, 3) -
                  inter.data(4, 4)) *
                     8;
    data(1, 0) = inter.data(1, 0) + inter.data(1, 1) + inter.data(1, 2) +
                 inter.data(1, 3) + inter.data(1, 4) - inter.data(2, 0) -
                 inter.data(2, 1) - inter.data(2, 2) - inter.data(2, 3) -
                 inter.data(2, 4) +
                 (inter.data(3, 0) + inter.data(3, 1) + inter.data(3, 2) +
                  inter.data(3, 3) + inter.data(3, 4) - inter.data(4, 0) -
                  inter.data(4, 1) - inter.data(4, 2) - inter.data(4, 3) -
                  inter.data(4, 4)) *
                     2;
    data(1, 1) = inter.data(1, 1) - inter.data(1, 2) - inter.data(2, 1) +
                 inter.data(2, 2) +
                 (inter.data(1, 3) - inter.data(1, 4) - inter.data(2, 3) +
                  inter.data(2, 4) + inter.data(3, 1) - inter.data(3, 2) -
                  inter.data(4, 1) + inter.data(4, 2)) *
                     2 +
                 (inter.data(3, 3) - inter.data(3, 4) - inter.data(4, 3) +
                  inter.data(4, 4)) *
                     4;
    data(1, 2) = inter.data(1, 1) + inter.data(1, 2) - inter.data(2, 1) -
                 inter.data(2, 2) +
                 (inter.data(3, 1) + inter.data(3, 2) - inter.data(4, 1) -
                  inter.data(4, 2)) *
                     2 +
                 (inter.data(1, 3) + inter.data(1, 4) - inter.data(2, 3) -
                  inter.data(2, 4)) *
                     4 +
                 (inter.data(3, 3) + inter.data(3, 4) - inter.data(4, 3) -
                  inter.data(4, 4)) *
                     8;
    data(1, 3) = inter.data(1, 1) - inter.data(1, 2) + inter.data(1, 5) -
                 inter.data(2, 1) + inter.data(2, 2) - inter.data(2, 5) +
                 (inter.data(3, 1) - inter.data(3, 2) + inter.data(3, 5) -
                  inter.data(4, 1) + inter.data(4, 2) - inter.data(4, 5)) *
                     2 +
                 (inter.data(1, 3) - inter.data(1, 4) - inter.data(2, 3) +
                  inter.data(2, 4)) *
                     8 +
                 (inter.data(3, 3) - inter.data(3, 4) - inter.data(4, 3) +
                  inter.data(4, 4)) *
                     16;
    data(2, 0) = inter.data(1, 0) + inter.data(1, 1) + inter.data(1, 2) +
                 inter.data(1, 3) + inter.data(1, 4) + inter.data(2, 0) +
                 inter.data(2, 1) + inter.data(2, 2) + inter.data(2, 3) +
                 inter.data(2, 4) +
                 (inter.data(3, 0) + inter.data(3, 1) + inter.data(3, 2) +
                  inter.data(3, 3) + inter.data(3, 4) + inter.data(4, 0) +
                  inter.data(4, 1) + inter.data(4, 2) + inter.data(4, 3) +
                  inter.data(4, 4)) *
                     4;
    data(2, 1) = inter.data(1, 1) - inter.data(1, 2) + inter.data(2, 1) -
                 inter.data(2, 2) +
                 (inter.data(1, 3) - inter.data(1, 4) + inter.data(2, 3) -
                  inter.data(2, 4)) *
                     2 +
                 (inter.data(3, 1) - inter.data(3, 2) + inter.data(4, 1) -
                  inter.data(4, 2)) *
                     4 +
                 (inter.data(3, 3) - inter.data(3, 4) + inter.data(4, 3) -
                  inter.data(4, 4)) *
                     8;
    data(2, 2) = inter.data(1, 1) + inter.data(1, 2) + inter.data(2, 1) +
                 inter.data(2, 2) +
                 (inter.data(1, 3) + inter.data(1, 4) + inter.data(2, 3) +
                  inter.data(2, 4) + inter.data(3, 1) + inter.data(3, 2) +
                  inter.data(4, 1) + inter.data(4, 2)) *
                     4 +
                 (inter.data(4, 3) + inter.data(4, 4) + inter.data(3, 3) +
                  inter.data(3, 4)) *
                     16;
    data(2, 3) = inter.data(1, 1) - inter.data(1, 2) + inter.data(1, 5) +
                 inter.data(2, 1) - inter.data(2, 2) + inter.data(2, 5) +
                 (inter.data(3, 1) - inter.data(3, 2) + inter.data(3, 5) +
                  inter.data(4, 1) - inter.data(4, 2) + inter.data(4, 5)) *
                     4 +
                 (inter.data(1, 3) - inter.data(1, 4) + inter.data(2, 3) -
                  inter.data(2, 4)) *
                     8 +
                 (inter.data(3, 3) - inter.data(3, 4) + inter.data(4, 3) -
                  inter.data(4, 4)) *
                     32;
    data(3, 0) = inter.data(1, 0) + inter.data(1, 1) + inter.data(1, 2) +
                 inter.data(1, 3) + inter.data(1, 4) - inter.data(2, 0) -
                 inter.data(2, 1) - inter.data(2, 2) - inter.data(2, 3) -
                 inter.data(2, 4) + inter.data(5, 0) + inter.data(5, 1) +
                 inter.data(5, 2) + inter.data(5, 3) + inter.data(5, 4) +
                 (inter.data(3, 0) + inter.data(3, 1) + inter.data(3, 2) +
                  inter.data(3, 3) + inter.data(3, 4) - inter.data(4, 0) -
                  inter.data(4, 1) - inter.data(4, 2) - inter.data(4, 3) -
                  inter.data(4, 4)) *
                     8;
    data(3, 1) = inter.data(1, 1) - inter.data(1, 2) - inter.data(2, 1) +
                 inter.data(2, 2) + inter.data(5, 1) - inter.data(5, 2) +
                 (inter.data(1, 3) - inter.data(1, 4) + inter.data(5, 3) -
                  inter.data(5, 4) - inter.data(2, 3) + inter.data(2, 4)) *
                     2 +
                 (inter.data(3, 1) - inter.data(3, 2) - inter.data(4, 1) +
                  inter.data(4, 2)) *
                     8 +
                 (inter.data(3, 3) - inter.data(3, 4) - inter.data(4, 3) +
                  inter.data(4, 4)) *
                     16;
    data(3, 2) = inter.data(1, 1) + inter.data(1, 2) - inter.data(2, 1) -
                 inter.data(2, 2) + inter.data(5, 1) + inter.data(5, 2) +
                 (inter.data(1, 3) + inter.data(1, 4) + inter.data(5, 3) +
                  inter.data(5, 4) - inter.data(2, 3) - inter.data(2, 4)) *
                     4 +
                 (inter.data(3, 1) + inter.data(3, 2) - inter.data(4, 1) -
                  inter.data(4, 2)) *
                     8 +
                 (inter.data(3, 3) + inter.data(3, 4) - inter.data(4, 3) -
                  inter.data(4, 4)) *
                     32;
    data(3, 3) = inter.data(1, 1) - inter.data(1, 2) + inter.data(1, 5) -
                 inter.data(2, 1) + inter.data(2, 2) - inter.data(2, 5) +
                 inter.data(5, 1) - inter.data(5, 2) + inter.data(5, 5) +
                 (inter.data(1, 3) - inter.data(1, 4) - inter.data(2, 3) +
                  inter.data(2, 4) + inter.data(3, 1) - inter.data(3, 2) +
                  inter.data(3, 5) - inter.data(4, 1) + inter.data(4, 2) -
                  inter.data(4, 5) + inter.data(5, 3) - inter.data(5, 4)) *
                     8 +
                 (inter.data(3, 3) - inter.data(3, 4) - inter.data(4, 3) +
                  inter.data(4, 4)) *
                     64;
  }
};

}  // namespace winograd
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_SRC_CONV2D_WINOGRAD_KERNELS_TILES_IMPL_H_

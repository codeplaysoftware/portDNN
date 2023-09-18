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
#ifndef PORTDNN_SRC_HELPERS_WINDOW_INDEX_H_
#define PORTDNN_SRC_HELPERS_WINDOW_INDEX_H_

#include "portdnn/helpers/macros.h"
#include "portdnn/helpers/non_deduced_type.h"
#include "portdnn/helpers/ratio.h"

namespace sycldnn {
namespace helpers {

template <typename Index>
struct WindowIndices {
  static_assert(std::is_integral<Index>::value,
                "WindowIndices Index type must be integral.");
  static_assert(std::is_signed<Index>::value,
                "WindowIndices Index type must be signed.");
  /** The index at which the window starts. */
  Index window_start;
  /** The index inside the window which is the first used. */
  Index filter_start;
};
/**
 * Get the index at which the window starts in the input tensor for the given
 * output index.
 *
 * NOTE: The window start index can be negative if the output index is less
 * than the padding value. An alternative approach to avoid this would be to
 * round any negative result to 0 and increase the filter start index
 * correspondingly.
 */
template <typename Index>
inline SNN_ALWAYS_INLINE WindowIndices<Index> in_window_from_output(
    Index const index, NonDeducedType<Index> const stride,
    NonDeducedType<Index> const pad) {
  WindowIndices<Index> const in_window{index * stride - pad, 0};
  return in_window;
}
/**
 * Get the index at which the window starts in the output tensor for the given
 * input index.
 *
 * NOTE: The padding here is expected to be the output padding.
 *   (pad_out = window - 1 - pad_in)
 * NOTE: The indices returned by this function will never be negative, however
 * the intermediate steps can be so the Index type must be signed.
 */
template <typename Index>
inline SNN_ALWAYS_INLINE WindowIndices<Index> out_window_from_input(
    Index const index, NonDeducedType<Index> const stride,
    NonDeducedType<Index> const pad) {
  Index const padded = index - pad;
  Index const window_start = round_ratio_up_above_zero(padded, stride);
  Index const filter_start = window_start * stride - padded;
  WindowIndices<Index> const out_window{window_start, filter_start};
  return out_window;
}
}  // namespace helpers
}  // namespace sycldnn
#endif  // PORTDNN_SRC_HELPERS_WINDOW_INDEX_H_

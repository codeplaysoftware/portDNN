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

#ifndef PORTDNN_SRC_ROI_ALIGN_KERNELS_H_
#define PORTDNN_SRC_ROI_ALIGN_KERNELS_H_

#include <CL/sycl.hpp>

#include "src/helpers/tensor_index.h"
#include "src/helpers/vector_io.h"
#include "src/helpers/vector_type.h"
#include "src/roi_align/operators_impl.h"

#include "portdnn/accessor_types.h"

#include "portdnn/helpers/minmax.h"

#include "portdnn/roi_align/params.h"

namespace sycldnn {

namespace roi_align {

template <typename T, template <typename> class Op>
struct interpolated_value;

template <typename T>
struct interpolated_value<T, MaxPool> {
  static T value(T w1, T w2, T w3, T w4, T v1, T v2, T v3, T v4) {
    return (cl::sycl::max(
        cl::sycl::max(cl::sycl::max(w1 * v1, w2 * v2), w3 * v3), w4 * v4));
  }
};

template <typename T>
struct interpolated_value<T, AveragePool> {
  static T value(T w1, T w2, T w3, T w4, T v1, T v2, T v3, T v4) {
    return (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  }
};

template <typename T, typename BatchIndicesT, typename Index,
          template <typename> class Op, bool IsUSM>
class RoiAlignOp {
  ReadMem<T const, IsUSM> in_data_;
  ReadMem<T const, IsUSM> roi_data_;
  ReadMem<BatchIndicesT const, IsUSM> batch_indices_data_;
  WriteMem<T, IsUSM> out_data_;
  RoiAlignParams params_;
  size_t const n_threads_;

  SNN_ALWAYS_INLINE T interpolate_bilinear(T const* in_ptr, Index height,
                                           Index width, T y, T x) const {
    if (y < T(-1) || y > static_cast<T>(height) || x < T(-1) ||
        x > static_cast<T>(width)) {
      return T(0);
    }

    y = cl::sycl::clamp(y, T(0), std::numeric_limits<T>::max());
    x = cl::sycl::clamp(x, T(0), std::numeric_limits<T>::max());

    Index y_low = cl::sycl::floor(y);
    Index x_low = cl::sycl::floor(x);
    Index y_high;
    Index x_high;

    if (y_low >= height - Index(1)) {
      y_high = y_low = height - Index(1);
      y = static_cast<T>(y_low);
    } else {
      y_high = y_low + Index(1);
    }

    if (x_low >= width - Index(1)) {
      x_high = x_low = width - Index(1);
      x = static_cast<T>(x_low);
    } else {
      x_high = x_low + Index(1);
    }

    T const ly = y - y_low;
    T const lx = x - x_low;
    T const hy = T(1) - ly, hx = T(1) - lx;

    auto const v1 = in_ptr[y_low * width + x_low];
    auto const v2 = in_ptr[y_low * width + x_high];
    auto const v3 = in_ptr[y_high * width + x_low];
    auto const v4 = in_ptr[y_high * width + x_high];
    auto const w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

    return interpolated_value<T, Op>::value(w1, w2, w3, w4, v1, v2, v3, v4);
  }

 public:
  SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) const {
    T const* in_ptr = in_data_.get_pointer();
    T const* roi_ptr = roi_data_.get_pointer();
    BatchIndicesT const* batch_indices_ptr = batch_indices_data_.get_pointer();
    T* out_ptr = out_data_.get_pointer();
    T const spatial_scale = static_cast<T>(params_.spatial_scale);

    // TODO: Optimize this loop; perform once per ROI instead of once per
    // element in the output.
    for (size_t index = item.get_linear_id(); index < n_threads_;
         index += item.get_range(0)) {
      Index const ow = index % params_.out_width;
      Index const oh = (index / params_.out_width) % params_.out_height;
      Index const c =
          ((index / params_.out_width / params_.out_height) % params_.channels);
      Index const n =
          index / params_.out_width / params_.out_height / params_.channels;

      T const* offset_roi_ptr = roi_ptr + n * params_.roi_cols;
      BatchIndicesT const roi_batch_idx = batch_indices_ptr[n];

      bool const is_output_half_pixel =
          (params_.coordinate_transformation_mode ==
           CoordinateTransformationMode::OUTPUT_HALF_PIXEL);
      T const roi_offset = is_output_half_pixel ? T(0) : T(0.5);
      T const roi_start_w = offset_roi_ptr[0] * spatial_scale - roi_offset;
      T const roi_start_h = offset_roi_ptr[1] * spatial_scale - roi_offset;
      T const roi_end_w = offset_roi_ptr[2] * spatial_scale - roi_offset;
      T const roi_end_h = offset_roi_ptr[3] * spatial_scale - roi_offset;

      T roi_width = roi_end_w - roi_start_w;
      T roi_height = roi_end_h - roi_start_h;
      if (is_output_half_pixel) {
        roi_width = cl::sycl::max(roi_width, T(1));
        roi_height = cl::sycl::max(roi_height, T(1));
      }

      T const bin_size_h =
          static_cast<T>(roi_height) / static_cast<T>(params_.out_height);
      T const bin_size_w =
          static_cast<T>(roi_width) / static_cast<T>(params_.out_width);

      T const* offset_in_ptr =
          in_ptr + static_cast<size_t>((roi_batch_idx * params_.channels + c) *
                                       params_.in_height * params_.in_width);

      Index const roi_bin_grid_h =
          (params_.sampling_ratio > 0)
              ? params_.sampling_ratio
              : cl::sycl::ceil(roi_height / params_.out_height);
      Index const roi_bin_grid_w =
          (params_.sampling_ratio > 0)
              ? params_.sampling_ratio
              : cl::sycl::ceil(roi_width / params_.out_width);

      Op<T> op{};
      for (Index iy = 0; iy < roi_bin_grid_h; iy++) {
        T const y = roi_start_h + oh * bin_size_h +
                    static_cast<T>(iy + T(.5)) * bin_size_h /
                        static_cast<T>(roi_bin_grid_h);
        for (Index ix = 0; ix < roi_bin_grid_w; ix++) {
          T const x = roi_start_w + ow * bin_size_w +
                      static_cast<T>(ix + T(.5)) * bin_size_w /
                          static_cast<T>(roi_bin_grid_w);

          auto const val = interpolate_bilinear(
              offset_in_ptr, params_.in_height, params_.in_width, y, x);
          op.accumulate(val);
        }
      }

      out_ptr[index] = op.value();
    }
  }

  RoiAlignOp(ReadMem<T const, IsUSM> in_data, ReadMem<T const, IsUSM> roi_data,
             ReadMem<BatchIndicesT const, IsUSM> batch_indices_data,
             WriteMem<T, IsUSM> out_data, RoiAlignParams const& rap,
             size_t n_threads)
      : in_data_(std::move(in_data)),
        roi_data_(std::move(roi_data)),
        batch_indices_data_(batch_indices_data),
        out_data_(std::move(out_data)),
        params_(rap),
        n_threads_(n_threads) {}
};

}  // namespace roi_align
}  // namespace sycldnn

#endif  // PORTDNN_SRC_ROI_ALIGN_KERNELS_H_

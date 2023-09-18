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

#ifndef PORTDNN_INCLUDE_COMPAT_POOLING_HPP
#define PORTDNN_INCLUDE_COMPAT_POOLING_HPP

#include <memory>
#include <vector>
#include "portdnn/compat/nan.h"
#include "portdnn/pooling/launch.h"
#include "portdnn/pooling/params.h"
#include "scaling.hpp"
#include "utils.hpp"

namespace sycldnn {
namespace compat {
/** The Pooling mode*/
enum class PoolingMode {
  POOLING_MAX = 0,
  POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
  POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
  POOLING_MAX_DETERMINISTIC

};

/**
 * class containing the padding, stride and window of the pooling
 * operation.
 */
class PoolingDescriptor {
  using Index_t = int;
  size_t nDims_;
  PoolingMode mode_;
  NanPropagation max_pooling_nan_opt_;
  std::vector<Index_t> window_dim_;
  std::vector<Index_t> padding_;
  std::vector<Index_t> stride_;

 public:
  PoolingDescriptor()
      : nDims_(2), window_dim_(2, 1), padding_(2, 0), stride_(2, 1) {
    SNN_COMPAT_ASSERT(nDims_ == 2,
                      "Cannot call method on non 2-D convolution desc.");
  }

  /** \return height of pooling window */
  Index_t getWindowH() const { return window_dim_[0]; };

  /** \return width of pooling window */
  Index_t getWindowW() const { return window_dim_[1]; };

  /** \return the stride across the height dimension */
  Index_t getStrideH() const { return stride_[0]; };

  /** \return the stride across the width dimension */
  Index_t getStrideW() const { return stride_[1]; };

  /** \return the padding across the height dimension */
  Index_t getPadH() const { return padding_[0]; };

  /** \return the padding across the width dimension */
  Index_t getPadW() const { return padding_[1]; };

  /** \return the pooling mode */
  PoolingMode getMode() const { return mode_; }
  /** \return the nan propagate option */
  NanPropagation getMaxPoolNanOpt() const { return max_pooling_nan_opt_; }

  /**
   * Sets the descriptor as a 2D pooling descriptor.
   * \param window_h Height of pooling window
   * \param window_w Width of pooling window
   * \param pad_h padding across the height dimension
   * \param pad_w padding across the width dimension
   * \param stride_h stride across the height dimension
   * \param stride_w stride across the width dimension
   * \param mode The PoolingMode to use
   * \param max_pooling_nan_opt The max pool nan propagation option
   * \return sycldnn::StatusCode
   */
  sycldnn::StatusCode set2d(
      int window_h, int window_w, int pad_h, int pad_w, int stride_h,
      int stride_w, PoolingMode mode,
      sycldnn::compat::NanPropagation max_pooling_nan_opt =
          sycldnn::compat::NanPropagation::NOT_PROPAGATE_NAN) {
    SNN_VALIDATE_PARAM(pad_h >= 0, "Invalid padding");
    SNN_VALIDATE_PARAM(pad_w >= 0, "Invalid padding");
    SNN_VALIDATE_PARAM(stride_h > 0, "Invalid stride");
    SNN_VALIDATE_PARAM(stride_w > 0, "Invalid stride");
    SNN_VALIDATE_PARAM(window_h >= 0, "Invalid window");
    SNN_VALIDATE_PARAM(window_w >= 0, "Invalid window");

    SNN_VALIDATE_PARAM(
        max_pooling_nan_opt ==
                sycldnn::compat::NanPropagation::NOT_PROPAGATE_NAN ||
            mode == PoolingMode::POOLING_MAX_DETERMINISTIC,
        "Only Max Pooling supports NaN propagation.");
    SNN_VALIDATE_PARAM(
        mode != PoolingMode::POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
        "portDNN only supports average pooling with padding excluded.");
    SNN_VALIDATE_PARAM(mode != PoolingMode::POOLING_MAX,
                       "portDNN only supports deterministic max pooling.");

    nDims_ = 2;
    padding_ = {pad_h, pad_w};
    stride_ = {stride_h, stride_w};
    window_dim_ = {window_h, window_w};
    mode_ = mode;
    max_pooling_nan_opt_ = max_pooling_nan_opt;
    return sycldnn::StatusCode::OK;
  }
};

/**
 * Sets the descriptor as a 2D pooling descriptor.
 * \param desc The descriptor to set the parameters of
 * \param window_h Height of pooling window
 * \param window_w Width of pooling window
 * \param pad_h padding across the height dimension
 * \param pad_w padding across the width dimension
 * \param stride_h stride across the height dimension
 * \param stride_w stride across the width dimension
 * \param mode The PoolingMode to use
 * \return sycldnn::StatusCode::OK if the descriptor was created successfully.
 *         sycldnn::StatusCode::InvalidParameter if one of the parameters had an
 *         incorrect value.
 */
sycldnn::StatusCode setPooling2dDescriptor(PoolingDescriptor& desc,
                                           PoolingMode mode,
                                           NanPropagation max_pooling_nan_opt,
                                           int window_w, int window_h,
                                           int pad_h, int pad_w, int stride_h,
                                           int stride_w) {
  return desc.set2d(window_w, window_h, pad_h, pad_w, stride_h, stride_w, mode,
                    max_pooling_nan_opt);
}

namespace internal {
/**
 * Converts the descriptor into a sycldnn::pooling::PoolingParams
 * \param xDesc Input descriptor.
 * \param yDesc Output descriptor.
 * \param poolDesc Pooling descriptor.
 * \return Converted sycldnn::pooling::PoolingParams
 */
inline sycldnn::pooling::PoolingParams descToSnnParams(
    const TensorDescriptor& xDesc, const TensorDescriptor& yDesc,
    const PoolingDescriptor& poolDesc) {
  sycldnn::pooling::PoolingParams pool_params{};

  int inN, inC, inH, inW;
  xDesc.get4dDescriptorDims(&inN, &inC, &inH, &inW);

  int outN, outC, outH, outW;
  yDesc.get4dDescriptorDims(&outN, &outC, &outH, &outW);

  pool_params.in_rows = inH;
  pool_params.in_cols = inW;

  pool_params.out_rows = outH;
  pool_params.out_cols = outW;

  pool_params.window_rows = poolDesc.getWindowH();
  pool_params.window_cols = poolDesc.getWindowW();

  pool_params.stride_rows = poolDesc.getStrideH();
  pool_params.stride_cols = poolDesc.getStrideW();

  pool_params.pad_rows = poolDesc.getPadH();
  pool_params.pad_cols = poolDesc.getPadW();

  pool_params.batch = inN;
  pool_params.channels = inC;

  return pool_params;
}
}  // namespace internal

/**
 * Performs the pooling forward operation.
 * \param handle The SNNHandle.
 * \param alpha Scaling factor used to blend y output.
 * \param xDesc Descriptor for the input tensor.
 * \param x Pointer to device memory for the input tensor.
 * \param poolDesc Descriptor for the pooling operation.
 * \param beta Scaling factor used to blend y output.
 * \param yDesc Descriptor for the output tensor.
 * \param y Pointer to device memory for the output.
 * \return SNNStatus for the operation.
 */
template <typename ValueT>
SNNStatus poolingForward(SNNHandle& handle, const PoolingDescriptor& poolDesc,
                         const ValueT* alpha, const TensorDescriptor& xDesc,
                         const void* x, const ValueT* beta,
                         const TensorDescriptor& yDesc, void* y) {
  ScalingParams scParams(handle.getBackend(), alpha, beta, yDesc.getSize(), y);
  SNNStatus poolingEvent;
  poolingEvent.event = scParams.constructMem(handle.getBackend());

  if (!scParams.isAlphaZero()) {
    sycldnn::pooling::PoolingParams poolingParams =
        internal::descToSnnParams(xDesc, yDesc, poolDesc);

    if (poolDesc.getMode() == PoolingMode::POOLING_MAX_DETERMINISTIC) {
      if (poolDesc.getMaxPoolNanOpt() == NanPropagation::NOT_PROPAGATE_NAN) {
        poolingEvent = sycldnn::pooling::launch<ValueT, sycldnn::pooling::Max,
                                                sycldnn::pooling::Forward>(
            static_cast<const ValueT*>(x), static_cast<ValueT*>(y),
            poolingParams, handle.getBackend(), {poolingEvent.event});
      } else {
        poolingEvent =
            sycldnn::pooling::launch<ValueT, sycldnn::pooling::MaxWithNan,
                                     sycldnn::pooling::Forward>(
                static_cast<const ValueT*>(x), static_cast<ValueT*>(y),
                poolingParams, handle.getBackend(), {poolingEvent.event});
      }
    } else {
      poolingEvent = sycldnn::pooling::launch<ValueT, sycldnn::pooling::Average,
                                              sycldnn::pooling::Forward>(
          static_cast<const ValueT*>(x), static_cast<ValueT*>(y), poolingParams,
          handle.getBackend(), {poolingEvent.event});
    }
  }

  return scParams.applyScaling(handle.getBackend(), {poolingEvent.event});
}
}  // namespace compat
}  // namespace sycldnn
#endif

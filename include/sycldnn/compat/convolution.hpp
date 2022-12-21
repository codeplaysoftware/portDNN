#ifndef SYCLDNN_INCLUDE_COMPAT_CONVOLUTION_HPP
#define SYCLDNN_INCLUDE_COMPAT_CONVOLUTION_HPP

#include <sycldnn/conv2d/algorithm.h>
#include <sycldnn/conv2d/launch.h>
#include <sycldnn/conv2d/params.h>
#include <sycldnn/conv2d/selector/direct_selector.h>
#include <sycldnn/conv2d/selector/im2col_selector.h>
#include <sycldnn/conv2d/selector/tiled_selector.h>
#include <sycldnn/conv2d/selector/winograd_selector.h>
#include "utils.hpp"

#include <memory>
#include <vector>

/**
 * \file
 * Wrapper API for convolution.
 */

namespace sycldnn {
namespace compat {

/**
 * class containing the padding, stride and dilation of the convolution
 * operation. Currently only 2D convolution is supported.
 */
class ConvolutionDescriptor {
  using Index_t = int;
  size_t nDims_;
  std::vector<Index_t> padding_;
  std::vector<Index_t> stride_;
  std::vector<Index_t> dilation_;

  void check2d() const {
    SNN_ASSERT(nDims_ == 2, "Cannot call method on non 4-D tensor desc.");
  }

 public:
  /** \return the stride across the _height dimension */
  Index_t getStrideH() const {
    check2d();
    return stride_[0];
  };

  /** \return the stride across the width dimension */
  Index_t getStrideW() const {
    check2d();
    return stride_[1];
  };

  /** \return the padding across the height dimension */
  Index_t getPadH() const {
    check2d();
    return padding_[0];
  };

  /** \return the padding across the width dimension */
  Index_t getPadW() const {
    check2d();
    return padding_[1];
  };

  /** \return the dilation across the height dimension */
  Index_t getDilationH() const {
    check2d();
    return dilation_[0];
  };

  /** \return the dilation across the width dimension */
  Index_t getDilationW() const {
    check2d();
    return dilation_[1];
  };

  /**
   * sets the descriptor as a 2D convolution descriptor.
   * \param pad_h padding across the height dimension
   * \param pad_w padding across the width dimension
   * \param stride_h stride across the height dimension
   * \param stride_w stride across the width dimension
   * \param dilation_h dilation across the height dimension
   * \param dilation_w dilation across the width dimension
   * \return sycldnn::StatusCode
   */
  sycldnn::StatusCode set2d(int pad_h, int pad_w, int stride_h, int stride_w,
                            int dilation_h, int dilation_w) {
    SNN_VALIDATE_PARAM(pad_h >= 0, "Invalid padding");
    SNN_VALIDATE_PARAM(pad_w >= 0, "Invalid padding");
    SNN_VALIDATE_PARAM(stride_h > 0, "Invalid stride");
    SNN_VALIDATE_PARAM(stride_w > 0, "Invalid stride");
    SNN_VALIDATE_PARAM(dilation_h >= 0, "Invalid dilation");
    SNN_VALIDATE_PARAM(dilation_w >= 0, "Invalid dilation");
    nDims_ = 2;
    padding_ = {pad_h, pad_w};
    stride_ = {stride_h, stride_w};
    dilation_ = {dilation_h, dilation_w};
    return sycldnn::StatusCode::OK;
  }
};

/**
 * descriptor for the filter in a convolution operation.
 * Currently only 4D filters are supported.
 */
class FilterDescriptor {
  using Index_t = int;
  size_t nDims_;
  std::vector<Index_t> dimensions_;
  sycldnn::FilterFormat format_;

  void check4d() const {
    SNN_ASSERT(nDims_ == 4, "Cannot call method on non 4-D tensor desc.");
  }

 public:
  /** \return number of output feature maps of a 4D filter descriptor */
  Index_t getK() const {
    check4d();
    return dimensions_[0];
  };

  /** \return number of input feature maps of a 4D filter descriptor */
  Index_t getC() const {
    check4d();
    return dimensions_[1];
  };

  /** \return filter height of a 4D filter descriptor */
  Index_t getH() const {
    check4d();
    return dimensions_[2];
  };

  /** \return filter width of a 4D filter descriptor */
  Index_t getW() const {
    check4d();
    return dimensions_[3];
  };

  /** \return data format for the filter */
  sycldnn::FilterFormat getFormat() const { return format_; };

  /**
   * sets the descriptor as a 4D filter descriptor
   * \param format Data format.
   * \param k Number of output feature maps.
   * \param c Number of input feature maps.
   * \param h Filter height.
   * \param w Filter width.
   * \return sycldnn::StatusCode with the success/fail result.*/
  sycldnn::StatusCode set4d(sycldnn::DataFormat format, int k, int c, int h,
                            int w) {
    SNN_VALIDATE_PARAM(k > 0,
                       "Non strictly positive dimensions are not supported.");
    SNN_VALIDATE_PARAM(c > 0,
                       "Non strictly positive dimensions are not supported.");
    SNN_VALIDATE_PARAM(h > 0,
                       "Non strictly positive dimensions are not supported.");
    SNN_VALIDATE_PARAM(w > 0,
                       "Non strictly positive dimensions are not supported.");
    nDims_ = 4;
    dimensions_ = {k, c, h, w};
    if (format == sycldnn::DataFormat::NCHW)
      format_ = sycldnn::FilterFormat::FCHW;
    else if (format == sycldnn::DataFormat::NHWC)
      format_ = sycldnn::FilterFormat::HWCF;
    else
      return sycldnn::StatusCode::InvalidParameter;
    return sycldnn::StatusCode::OK;
  }
};

namespace internal {

/**
 * Converts the descriptor into a sycldnn::conv2d::Conv2DParam
 * \param xDesc Input descriptor.
 * \param yDesc Output descriptor.
 * \param wDesc Filter descriptor.
 * \param convDesc Convolution descriptor.
 * \return Converted sycldnn::conv2d::Conv2DParam
 */
inline sycldnn::conv2d::Conv2DParams descToSnnParams(
    const TensorDescriptor& xDesc, const TensorDescriptor& yDesc,
    const FilterDescriptor& wDesc, const ConvolutionDescriptor& convDesc) {
  sycldnn::conv2d::Conv2DParams conv_params{};
  conv_params.channels = xDesc.getC();
  conv_params.features = wDesc.getK();
  conv_params.batch = xDesc.getN();

  conv_params.in_rows = xDesc.getH();
  conv_params.in_cols = xDesc.getW();

  conv_params.window_rows = wDesc.getH();
  conv_params.window_cols = wDesc.getW();
  conv_params.stride_rows = convDesc.getStrideH();
  conv_params.stride_cols = convDesc.getStrideW();

  conv_params.out_rows = yDesc.getH();
  conv_params.out_cols = yDesc.getW();
  conv_params.pad_rows = convDesc.getPadH();
  conv_params.pad_cols = convDesc.getPadW();
  conv_params.dilation_rows = convDesc.getDilationH();
  conv_params.dilation_cols = convDesc.getDilationW();

  conv_params.filter_format = wDesc.getFormat();
  conv_params.input_format = xDesc.getFormat();
  return conv_params;
}

/** Returns the constant selector for a given algorithm
 * \param algo The algorithm
 * \return The constant selector
 */
inline std::unique_ptr<conv2d::Selector> getSelector(conv2d::Algorithm algo) {
  using algo_t = conv2d::Algorithm;
  switch (algo) {
    case algo_t::Direct:
      return std::make_unique<conv2d::DirectSelector>();
    default:
      return nullptr;
  }
}

}  // namespace internal

/**
 * Computes the dimension of the output descriptor.
 * \param desc Descriptor for the convolution operation.
 * \param in Descriptor for the input tensor.
 * \param filt Descriptor for the filter tensor.
 * \param n Output, pointer to the resulting batch size.
 * \param c Output, poitner to the resulting number of channels.
 * \param h Output, pointer to the resulting height.
 * \param w Output, pointer to the resulting width.
 * \return SNNStaus::OK or SNNStatus::InvalidParameter
 */
inline sycldnn::StatusCode getConvolution2dForwardOutputDim(
    const ConvolutionDescriptor& desc, const TensorDescriptor& in,
    const FilterDescriptor& filt, int* n, int* c, int* h, int* w) {
  SNN_VALIDATE_PARAM(n != nullptr, "Output pointer cannot be null");
  SNN_VALIDATE_PARAM(c != nullptr, "Output pointer cannot be null");
  SNN_VALIDATE_PARAM(h != nullptr, "Output pointer cannot be null");
  SNN_VALIDATE_PARAM(w != nullptr, "Output pointer cannot be null");
  using Index_t = int;
  *n = in.getN();
  *c = filt.getK();
  auto computeNewDim = [](Index_t inputDim, Index_t filterDim, Index_t pad,
                          Index_t dilation, Index_t convolutionStride) {
    return 1 + (inputDim + 2 * pad - (((filterDim - 1) * dilation) + 1)) /
                   convolutionStride;
  };
  *h = computeNewDim(in.getH(), filt.getH(), desc.getPadH(),
                     desc.getDilationH(), desc.getStrideH());
  *w = computeNewDim(in.getW(), filt.getW(), desc.getPadW(),
                     desc.getDilationW(), desc.getStrideW());
  return sycldnn::StatusCode::OK;
}

/**
 * Performs the convolution forward operation.
 * \param handle The SNNHandle.
 * \param alpha Scaling factor, currently unused.
 * \param xDesc Descriptor for the input tensor.
 * \param x Pointer to device memory for the input tensor.
 * \param wDesc Descriptor for the filter.
 * \param w Pointer to device memory for the filter.
 * \param convDesc Descriptor for the convolution operation.
 * \param algo Convolution algorithm to be employed.
 * \param workSpace Pointer to device scratchpad memory, currently unused.
 * \param workSpaceSizeInBytes size of the scratchpad memory, currentl unused.
 * \param beta Scaling factor, currently unused.
 * \param yDesc Descriptor for the output tensor, its dimension can be obtained
 * with getConvolution2dForwardOutputDim.
 * \param y Pointer to device memory for the output.
 * \return SNNStatus for the operation.
 */
template <typename ValueT = float>
SNNStatus convolutionForward(SNNHandle& handle, const void* alpha,
                             const TensorDescriptor& xDesc, const void* x,
                             const FilterDescriptor& wDesc, const void* w,
                             const ConvolutionDescriptor& convDesc,
                             conv2d::Algorithm algo, void* workSpace,
                             size_t workSpaceSizeInBytes, const void* beta,
                             const TensorDescriptor& yDesc, void* y) {
  SNN_UNUSED_VAR(alpha);
  SNN_UNUSED_VAR(beta);

  sycldnn::conv2d::Conv2DParams conv1_params =
      internal::descToSnnParams(xDesc, yDesc, wDesc, convDesc);

  std::unique_ptr<conv2d::Selector> selector = internal::getSelector(algo);
  SNN_VALIDATE_PARAM(selector != nullptr, "Unsupported algorithm");
  return sycldnn::conv2d::launch<ValueT, sycldnn::conv2d::conv_type::Forward>(
      static_cast<const ValueT*>(x), static_cast<const ValueT*>(w),
      static_cast<ValueT*>(y), conv1_params, *selector, handle.getBackend(),
      static_cast<ValueT*>(workSpace), workSpaceSizeInBytes, {});
}

}  // namespace compat

}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_COMPAT_CONVOLUTION_HPP

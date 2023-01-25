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

#ifndef SYCLDNN_INCLUDE_COMPAT_CONVOLUTION_HPP
#define SYCLDNN_INCLUDE_COMPAT_CONVOLUTION_HPP

#include "utils.hpp"

#include "sycldnn/conv2d/algorithm.h"
#include "sycldnn/conv2d/launch.h"
#include "sycldnn/conv2d/params.h"
#include "sycldnn/conv2d/selector/direct_selector.h"
#include "sycldnn/conv2d/selector/im2col_selector.h"
#include "sycldnn/conv2d/selector/tiled_selector.h"
#include "sycldnn/conv2d/selector/winograd_selector.h"

#include <memory>
#include <vector>

/**
 * \file
 * Wrapper API for convolution.
 */

namespace sycldnn {
namespace compat {

/** The Convolution mode */
enum class ConvolutionMode {
  CONVOLUTION = 0,    // Do a convolution operation, applying the filter to the
                      // input. Currently not supported.
  CROSS_CORRELATION,  // Do a cross-correlation operation, applying the rotated
                      // filter to the images.
};

/**
 * class containing the padding, stride and dilation of the convolution
 * operation. Currently only 2D convolution is supported.
 */
class ConvolutionDescriptor {
  using Index_t = int;
  /** Number of convlolution dimensions (default to 2)*/
  size_t nDims_;

  /** Vector containing size of padding of descriptor for each dimension */
  std::vector<Index_t> padding_;
  /** Vector containing size of stride of descriptor for each dimension */
  std::vector<Index_t> stride_;
  /** Vector containing size of dilation of descriptor for each dimension */
  std::vector<Index_t> dilation_;
  /** Enum of convolution type of descriptor \ref
   * sycldnn::compat::ConvolutionMode */
  ConvolutionMode mode_;

 public:
  /**
   * Default constructor
   */
  ConvolutionDescriptor()
      : nDims_(2), padding_(2, 0), stride_(2, 1), dilation_(2, 1) {
    SNN_COMPAT_ASSERT(nDims_ == 2,
                      "Cannot call method on non 2-D convolution desc.");
  }
  /** \return the number of spatial dimensions */
  size_t getNumDims() const { return nDims_; }

  /** \return an std::vector<int> containing the padding values across the
   * spatial dimensions */
  std::vector<Index_t> getPadding() const { return padding_; }

  /** \return an std::vector<int> containing the stride values across the
   * spatial dimensions */
  std::vector<Index_t> getStride() const { return stride_; }

  /** \return an std::vector<int> containing the dilation values across the
   * spatial dimensions */
  std::vector<Index_t> getDilation() const { return dilation_; }

  /** \return the convolution mode */
  ConvolutionMode getMode() const { return mode_; }

  /** \return the stride across the height dimension */
  Index_t getStrideH() const { return stride_[0]; };

  /** \return the stride across the width dimension */
  Index_t getStrideW() const { return stride_[1]; };

  /** \return the padding across the height dimension */
  Index_t getPadH() const { return padding_[0]; };

  /** \return the padding across the width dimension */
  Index_t getPadW() const { return padding_[1]; };

  /** \return the dilation across the height dimension */
  Index_t getDilationH() const { return dilation_[0]; };

  /** \return the dilation across the width dimension */
  Index_t getDilationW() const { return dilation_[1]; };

  /**
   * Sets the descriptor as a 2D convolution descriptor.
   * \param pad_h       Padding across the height dimension
   * \param pad_w       Padding across the width dimension
   * \param stride_h    Stride across the height dimension
   * \param stride_w    Stride across the width dimension
   * \param dilation_h  Dilation across the height dimension
   * \param dilation_w  Dilation across the width dimension
   * \param mode        The ConvolutionMode to use
   * \return            sycldnn::StatusCode
   */
  sycldnn::StatusCode set2d(
      int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h,
      int dilation_w,
      ConvolutionMode mode = ConvolutionMode::CROSS_CORRELATION) {
    SNN_VALIDATE_PARAM(pad_h >= 0, "Invalid padding");
    SNN_VALIDATE_PARAM(pad_w >= 0, "Invalid padding");
    SNN_VALIDATE_PARAM(stride_h > 0, "Invalid stride");
    SNN_VALIDATE_PARAM(stride_w > 0, "Invalid stride");
    SNN_VALIDATE_PARAM(dilation_h >= 0, "Invalid dilation");
    SNN_VALIDATE_PARAM(dilation_w >= 0, "Invalid dilation");
    SNN_VALIDATE_PARAM(
        mode == ConvolutionMode::CROSS_CORRELATION,
        "Only ConvolutionMode::CROSS_CORRELATION is currently supported");
    nDims_ = 2;
    padding_ = {pad_h, pad_w};
    stride_ = {stride_h, stride_w};
    dilation_ = {dilation_h, dilation_w};
    mode_ = mode;
    return sycldnn::StatusCode::OK;
  }

  /**
   * Sets the descriptor as an N-dimensional convolution descriptor.
   *
   * \param pads              Vector containing the padding values across each
   *                          spatial dimension
   * \param strides           Vector containing the stride values across each
   *                          spatial dimension
   * \param dilations         Vector containing the dilation values
   *                          across each spatial dimension
   * \param mode              The sycldnn::compat::ConvolutionMode to use
   * \return                  sycldnn::StatusCode
   */
  sycldnn::StatusCode setNd(
      const std::vector<int>& pads, const std::vector<int>& strides,
      const std::vector<int>& dilations,
      ConvolutionMode mode = ConvolutionMode::CROSS_CORRELATION) {
    const bool num_dims_match =
        pads.size() == strides.size() && pads.size() == dilations.size();
    SNN_VALIDATE_PARAM(
        num_dims_match,
        "Pads, strides and dilations must have the same number of elements");
    for (auto pad : pads) {
      SNN_VALIDATE_PARAM(pad >= 0, "Invalid padding");
    }
    for (auto stride : strides) {
      SNN_VALIDATE_PARAM(stride > 0, "Invalid stride");
    }
    for (auto dilation : dilations) {
      SNN_VALIDATE_PARAM(dilation >= 1, "Invalid dilation");
    }
    SNN_VALIDATE_PARAM(
        mode == ConvolutionMode::CROSS_CORRELATION,
        "Only ConvolutionMode::CROSS_CORRELATION is currently supported");
    nDims_ = pads.size();
    padding_ = pads;
    stride_ = strides;
    dilation_ = dilations;
    mode_ = mode;
    return sycldnn::StatusCode::OK;
  }
};

/**
 * Sets the descriptor as a 2D convolution descriptor.
 * \param desc        The descriptor to set the parameters of
 * \param pad_h       padding across the height dimension
 * \param pad_w       padding across the width dimension
 * \param stride_h    stride across the height dimension
 * \param stride_w    stride across the width dimension
 * \param dilation_h  dilation across the height dimension
 * \param dilation_w  dilation across the width dimension
 * \param mode        The ConvolutionMode to use
 * \return            sycldnn::StatusCode::OK if the descriptor was created
 *                    successfully. sycldnn::StatusCode::InvalidParameter if one
 *                    of the parameters had an incorrect value.
 */
sycldnn::StatusCode setConvolution2dDescriptor(ConvolutionDescriptor& desc,
                                               int pad_h, int pad_w,
                                               int stride_h, int stride_w,
                                               int dilation_h, int dilation_w,
                                               ConvolutionMode mode) {
  return desc.set2d(pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
                    mode);
}

/**
 * Sets the descriptor as an N-dimensional convolution descriptor.
 *
 * \param desc              The descriptor to set the parameters of
 * \param num_spatial_dims  The number of Convolution dimensions
 * \param pads              Vector containing the padding values across each
 *                          spatial dimension
 * \param strides           Vector containing the stride values across each
 *                          spatial dimension
 * \param dilations         Vector containing the dilation values
 *                          across each spatial dimension
 * \param mode              The sycldnn::compat::ConvolutionMode to use
 * \return                  sycldnn::StatusCode::OK if the descriptor was
 *                          created successfully.
 *                          sycldnn::StatusCode::InvalidParameter if
 *                          num_spatial_dims != 2 (currently only 2D
 *                          Convolution is supported)
 */
sycldnn::StatusCode setConvolutionNdDescriptor(
    ConvolutionDescriptor& desc, int num_spatial_dims, const int* pads,
    const int* strides, const int* dilations, ConvolutionMode mode) {
  if (num_spatial_dims != 2) {
    return sycldnn::StatusCode::InvalidParameter;
  }

  const std::vector<int> pads_vec(pads, pads + num_spatial_dims),
      strides_vec(strides, strides + num_spatial_dims),
      dilations_vec(dilations, dilations + num_spatial_dims);

  return desc.setNd(pads_vec, strides_vec, dilations_vec, mode);
}

/**
 * Descriptor for the filter in a convolution operation.
 * currently only 4D filters are supported.
 */
class FilterDescriptor : public DescriptorBase {
  sycldnn::FilterFormat format_;

 public:
  /**
   * Default constructor, dimensions set to be 4
   */
  FilterDescriptor() : DescriptorBase() {}
  /**
   * Constructor which takes in number of dimensions
   * @param nDims   Number of dimensions to initialize the descriptor to
   */
  FilterDescriptor(size_t nDims) : DescriptorBase(nDims) {}

  /** \return total size of the 4D filter (number of elements)*/
  size_t getSize() const {
    return std::accumulate(dims_.begin(), dims_.end(), 1,
                           std::multiplies<size_t>());
  }

  /** \copydoc DescriptorBase::set4d */
  sycldnn::StatusCode set4d(sycldnn::DataFormat format, int dim0, int dim1,
                            int dim2, int dim3) override final {
    SNN_VALIDATE_PARAM(dim0 > 0 && dim1 > 0 && dim2 > 0 && dim3 > 0,
                       "Non strictly positive dimensions are not supported.");
    nDims_ = 4;
    if (format == sycldnn::DataFormat::NCHW) {
      format_ = sycldnn::FilterFormat::FCHW;
      dims_ = {dim0, dim1, dim2, dim3};

    } else if (format == sycldnn::DataFormat::NHWC) {
      format_ = sycldnn::FilterFormat::HWCF;
      dims_ = {dim2, dim3, dim1, dim0};

    } else {
      return sycldnn::StatusCode::InvalidParameter;
    }
    return sycldnn::StatusCode::OK;
  }

  /** This function queries the NCHW params of the previously initialized
   * descriptor object.
   * \param k   Output number of output feature maps.
   * \param c   Output number of input feature maps per image.
   * \param h   Output height of each feature map.
   * \param w   Output width of each feature map.
   * \return    sycldnn::StatusCode::OK or sycldnn::StatusCode::InvalidParameter
   */
  sycldnn::StatusCode get4dDescriptorDims(int* k, int* c, int* h, int* w) {
    SNN_VALIDATE_PARAM(k != nullptr, "Output pointer cannot be null");
    SNN_VALIDATE_PARAM(c != nullptr, "Output pointer cannot be null");
    SNN_VALIDATE_PARAM(h != nullptr, "Output pointer cannot be null");
    SNN_VALIDATE_PARAM(w != nullptr, "Output pointer cannot be null");

    if (format_ == sycldnn::FilterFormat::FCHW) {
      *k = dims_[0];
      *c = dims_[1];
      *h = dims_[2];
      *w = dims_[3];
    } else if (format_ == sycldnn::FilterFormat::HWCF) {
      *h = dims_[0];
      *w = dims_[1];
      *c = dims_[2];
      *k = dims_[3];
    } else {
      return sycldnn::StatusCode::InvalidParameter;
    }
    return sycldnn::StatusCode::OK;
  }

  /** This function queries the parameters of the previously initialized
   * descriptor object.
   * \param dataType  Output data type.
   * \param format    Format of the filter descriptor KCHW or KHWC
   * \param k   Output number of output feature maps.
   * \param c   Output number of input feature maps per image.
   * \param h         Output height of each feature map.
   * \param w         Output width of each feature map.
   * \return          sycldnn::StatusCode::OK or
   *                  sycldnn::StatusCode::InvalidParameter
   */
  sycldnn::StatusCode getFilter4dDescriptor(SNNDataType* dataType,
                                            sycldnn::FilterFormat* format,
                                            int* k, int* c, int* h, int* w) {
    SNN_VALIDATE_PARAM(dataType != nullptr, "Output pointer cannot be null");
    SNN_VALIDATE_PARAM(format != nullptr, "Output pointer cannot be null");
    *dataType = SNNDataType::SNN_FLOAT;
    *format = format_;
    return get4dDescriptorDims(k, c, h, w);
  }
};

/** This function queries the parameters of the previously initialized
 * descriptor object.
 * \param dataType    Output data type.
 * \param filterDesc  Input a previously created filter descriptor.
 * \param format      Format of the filter descriptor KCHW or KHWC
 * \param k           Output number of output feature maps.
 * \param c           Output number of input feature maps per image.
 * \param h           Output height of each feature map.
 * \param w           Output width of each feature map.
 * \return            sycldnn::StatusCode::OK or
 *                    sycldnn::StatusCode::InvalidParameter
 */
sycldnn::StatusCode getFilter4dDescriptor(FilterDescriptor filterDesc,
                                          SNNDataType* dataType,
                                          sycldnn::FilterFormat* format, int* k,
                                          int* c, int* h, int* w) {
  return filterDesc.getFilter4dDescriptor(dataType, format, k, c, h, w);
}

namespace internal {

/**
 * Converts the descriptor into a sycldnn::conv2d::Conv2DParam
 * \param xDesc     Input descriptor.
 * \param yDesc     Output descriptor.
 * \param wDesc     Filter descriptor.
 * \param convDesc  Convolution descriptor.
 * \return          Converted sycldnn::conv2d::Conv2DParam
 */
inline sycldnn::conv2d::Conv2DParams descToSnnParams(
    const TensorDescriptor& xDesc, const TensorDescriptor& yDesc,
    const FilterDescriptor& wDesc, const ConvolutionDescriptor& convDesc) {
  sycldnn::conv2d::Conv2DParams conv_params{};

  SNNDataType descDataType;
  int inN;
  int inC;
  int inH;
  int inW;
  int inStrideN;
  int inStrideC;
  int inStrideH;
  int inStrideW;
  getTensor4dDescriptor(xDesc, &descDataType, &inN, &inC, &inH, &inW,
                        &inStrideN, &inStrideC, &inStrideH, &inStrideW);

  sycldnn::FilterFormat format;
  int filterK;
  int filterC;
  int filterH;
  int filterW;
  getFilter4dDescriptor(wDesc, &descDataType, &format, &filterK, &filterC,
                        &filterH, &filterW);

  int outN;
  int outC;
  int outH;
  int outW;
  int outStrideN;
  int outStrideC;
  int outStrideH;
  int outStrideW;
  getTensor4dDescriptor(yDesc, &descDataType, &outN, &outC, &outH, &outW,
                        &outStrideN, &outStrideC, &outStrideH, &outStrideW);

  conv_params.channels = inC;
  conv_params.features = filterK;
  conv_params.batch = inN;

  conv_params.in_rows = inH;
  conv_params.in_cols = inW;

  conv_params.window_rows = filterH;
  conv_params.window_cols = filterW;
  conv_params.stride_rows = convDesc.getStrideH();
  conv_params.stride_cols = convDesc.getStrideW();

  conv_params.out_rows = outH;
  conv_params.out_cols = outW;
  conv_params.pad_rows = convDesc.getPadH();
  conv_params.pad_cols = convDesc.getPadW();
  conv_params.dilation_rows = convDesc.getDilationH();
  conv_params.dilation_cols = convDesc.getDilationW();

  conv_params.filter_format = format;
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
 * \param desc  Descriptor for the convolution operation.
 * \param in    Descriptor for the input tensor.
 * \param filt  Descriptor for the filter tensor.
 * \param n     Output, pointer to the resulting batch size.
 * \param c     Output, poitner to the resulting number of channels.
 * \param h     Output, pointer to the resulting height.
 * \param w     Output, pointer to the resulting width.
 * \return      sycldnn::StatusCode::OK or
 * sycldnn::StatusCode::InvalidParameter
 */
inline sycldnn::StatusCode getConvolution2dForwardOutputDim(
    const ConvolutionDescriptor& desc, const TensorDescriptor& in,
    const FilterDescriptor& filt, int* n, int* c, int* h, int* w) {
  SNN_VALIDATE_PARAM(n != nullptr, "Output pointer cannot be null");
  SNN_VALIDATE_PARAM(c != nullptr, "Output pointer cannot be null");
  SNN_VALIDATE_PARAM(h != nullptr, "Output pointer cannot be null");
  SNN_VALIDATE_PARAM(w != nullptr, "Output pointer cannot be null");
  using Index_t = int;
  auto computeNewDim = [](Index_t inputDim, Index_t filterDim, Index_t pad,
                          Index_t dilation, Index_t convolutionStride) {
    return 1 + (inputDim + 2 * pad - (((filterDim - 1) * dilation) + 1)) /
                   convolutionStride;
  };
  SNNDataType descDataType;
  int inN;
  int inC;
  int inH;
  int inW;
  int inStrideN;
  int inStrideC;
  int inStrideH;
  int inStrideW;
  getTensor4dDescriptor(in, &descDataType, &inN, &inC, &inH, &inW, &inStrideN,
                        &inStrideC, &inStrideH, &inStrideW);

  sycldnn::FilterFormat format;
  int filterK;
  int filterC;
  int filterH;
  int filterW;
  getFilter4dDescriptor(filt, &descDataType, &format, &filterK, &filterC,
                        &filterH, &filterW);
  *n = inN;
  *c = filterK;
  *h = computeNewDim(inH, filterH, desc.getPadH(), desc.getDilationH(),
                     desc.getStrideH());
  *w = computeNewDim(inW, filterW, desc.getPadW(), desc.getDilationW(),
                     desc.getStrideW());
  return sycldnn::StatusCode::OK;
}

/**
 * Performs the convolution forward operation.
 * \param handle                The SNNHandle.
 * \param alpha                 Scaling factor, currently unused.
 * \param xDesc                 Descriptor for the input tensor.
 * \param x                     Pointer to device memory for the input tensor.
 * \param wDesc                 Descriptor for the filter.
 * \param w                     Pointer to device memory for the filter.
 * \param convDesc              Descriptor for the convolution operation.
 * \param algo                  Convolution algorithm to be employed.
 * \param workSpace             Pointer to device scratchpad memory, currently
 *                              unused.
 * \param workSpaceSizeInBytes  Size of the scratchpad memory, currentl
 *                              unused.
 * \param beta                  Scaling factor, currently unused.
 * \param yDesc                 Descriptor for the output tensor, its dimension
 *                              can be obtained with
 *                              getConvolution2dForwardOutputDim.
 * \param y                     Pointer to device memory for the output.
 * \return                      SNNStatus for the operation.
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

/**
 * Performs the convolution backward data operation.
 * \param handle The SNNHandle.
 * \param alpha Scaling factor, currently unused.
 * \param wDesc Descriptor for the filter.
 * \param w Pointer to device memory for the filter.
 * \param dyDesc Descriptor for the input differential tensor.
 * \param dy Pointer to device memory for the input differential tensor.
 * \param convDesc Descriptor for the convolution operation.
 * \param algo Convolution algorithm to be employed.
 * \param workSpace Pointer to device scratchpad memory, currently unused.
 * \param workSpaceSizeInBytes size of the scratchpad memory, currently unused.
 * \param beta Scaling factor, currently unused.
 * \param dxDesc Descriptor for the output tensor, its dimension can be obtained
 * with getConvolution2dForwardOutputDim.
 * \param dx Pointer to device memory for the output.
 * \return SNNStatus for the operation.
 */
template <typename ValueT = float>
SNNStatus convolutionBackwardData(SNNHandle& handle, const void* alpha,
                                  const FilterDescriptor wDesc, const void* w,
                                  const TensorDescriptor dyDesc, const void* dy,
                                  const ConvolutionDescriptor convDesc,
                                  conv2d::Algorithm algo, void* workSpace,
                                  size_t workSpaceSizeInBytes, const void* beta,
                                  const TensorDescriptor dxDesc, void* dx) {
  SNN_UNUSED_VAR(alpha);
  SNN_UNUSED_VAR(beta);

  sycldnn::conv2d::Conv2DParams conv1_params =
      internal::descToSnnParams(dxDesc, dyDesc, wDesc, convDesc);

  std::unique_ptr<conv2d::Selector> selector = internal::getSelector(algo);
  SNN_VALIDATE_PARAM(selector != nullptr, "Unsupported algorithm");
  return sycldnn::conv2d::launch<ValueT,
                                 sycldnn::conv2d::conv_type::InputBackprop>(
      static_cast<const ValueT*>(dy), static_cast<const ValueT*>(w),
      static_cast<ValueT*>(dx), conv1_params, *selector, handle.getBackend(),
      static_cast<ValueT*>(workSpace), workSpaceSizeInBytes, {});
}

/**
 * Performs the convolution backward filter operation.
 * \param handle The SNNHandle.
 * \param alpha Scaling factor, currently unused.
 * \param xDesc Descriptor for the input tensor.
 * \param x Pointer to device memory for the input tensor.
 * \param dyDesc Descriptor for the input differential tensor.
 * \param dy Pointer to device memory for the input differential tensor.
 * \param convDesc Descriptor for the convolution operation.
 * \param algo Convolution algorithm to be employed.
 * \param workSpace Pointer to device scratchpad memory, currently unused.
 * \param workSpaceSizeInBytes size of the scratchpad memory, currently unused.
 * \param beta Scaling factor, currently unused.
 * \param dwDesc Descriptor for the filter gradient.
 * \param dw Pointer to device memory for the filter gradient.
 * \return SNNStatus for the operation.
 */
template <typename ValueT = float>
SNNStatus convolutionBackwardFilter(
    SNNHandle& handle, const void* alpha, const TensorDescriptor& xDesc,
    const void* x, const TensorDescriptor& dyDesc, const void* dy,
    const ConvolutionDescriptor& convDesc, conv2d::Algorithm algo,
    void* workSpace, size_t workSpaceSizeInBytes, const void* beta,
    const FilterDescriptor& dwDesc, void* dw) {
  SNN_UNUSED_VAR(alpha);
  SNN_UNUSED_VAR(beta);

  sycldnn::conv2d::Conv2DParams conv1_params =
      internal::descToSnnParams(xDesc, dyDesc, dwDesc, convDesc);

  std::unique_ptr<conv2d::Selector> selector = internal::getSelector(algo);
  SNN_VALIDATE_PARAM(selector != nullptr, "Unsupported algorithm");
  return sycldnn::conv2d::launch<ValueT,
                                 sycldnn::conv2d::conv_type::FilterBackprop>(
      static_cast<const ValueT*>(x), static_cast<const ValueT*>(dy),
      static_cast<ValueT*>(dw), conv1_params, *selector, handle.getBackend(),
      static_cast<ValueT*>(workSpace), workSpaceSizeInBytes, {});
}

/** This function queries the parameters of the previously initialized
 * descriptor object.
 * \param filterDesc  Input a previously initialized filter descriptor.
 * \param dataType    Output data type.
 * \param format      Format of the filter descriptor KCHW or KHWC
 * \param k           Output number of output feature maps.
 * \param c           Output number of input feature maps per image.
 * \param h           Output height of each feature map.
 * \param w           Output width of each feature map.
 * \return            sycldnn::StatusCode::OK or
 *                    sycldnn::StatusCode::InvalidParameter
 */
sycldnn::StatusCode setFilter4dDescriptor(FilterDescriptor& filterDesc,
                                          SNNDataType dataType,
                                          sycldnn::DataFormat format, int k,
                                          int c, int h, int w) {
  SNN_UNUSED_VAR(dataType);
  return filterDesc.set4d(format, k, c, h, w);
}

}  // namespace compat

}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_COMPAT_CONVOLUTION_HPP

#ifndef SYCLDNN_INCLUDE_COMPAT_UTILS_HPP
#define SYCLDNN_INCLUDE_COMPAT_UTILS_HPP
#include <sycldnn/backend/snn_usm_backend.h>
#include <sycldnn/data_format.h>
#include <sycldnn/filter_format.h>
#include <sycldnn/status.h>
#include <vector>

/**
 * \file
 * Contains descriptor and helper classes used in the rest of the compat API.
 */

namespace sycldnn {

namespace compat {

/**
 * Wrapper around the sycl-dnn backends.
 */
class SNNHandle {
  sycl::queue q_;
  backend::SNNUSMBackend b_;

 public:
  /** Constructs the SNNHandle for a SNNUSMBackend.
   * \param d device selector, defaults to sycl::default_selector
   * \param prop a property list for the queue, default to setting
   * sycl::property::queue::in_order
   */
  SNNHandle(
      const sycl::device_selector& d = sycl::default_selector(),
      const sycl::property_list& prop = {sycl::property::queue::in_order()})
      : q_(d, prop), b_(q_) {}

  /** \return the SNNBackend */
  backend::SNNUSMBackend& getBackend() { return b_; }

  /** \return the sycl::queue */
  sycl::queue getQueue() { return q_; }
};

/**
 * class that describes the dimensions, strides, and data format for a tensor.
 * Currently only 4D tensors are supported, with the NCHW or NHWC formats.
 */
class TensorDescriptor {
  using Index_t = int;
  size_t nDims_;
  std::vector<Index_t> dims_;
  std::vector<Index_t> stride_;
  sycldnn::DataFormat format_;
  void check4d() const {
    SNN_ASSERT(nDims_ == 4, "Cannot call method on non 4-D tensor desc.");
  }

 public:
  /** \return batch size of a 4D tensor. */
  Index_t getN() const {
    check4d();
    return dims_[0];
  }

  /** \return number of channels of a 4D tensor. */
  Index_t getC() const {
    check4d();
    return dims_[1];
  }

  /** \return height of a 4D tensor. */
  Index_t getH() const {
    check4d();
    return dims_[2];
  }

  /** \return width of a 4D tensor. */
  Index_t getW() const {
    check4d();
    return dims_[3];
  }

  /** \return total size of the tensor (number of elements)*/
  size_t getSize() const {
    check4d();
    return dims_[0] * dims_[1] * dims_[2] * dims_[3];
  }

  /** \return data format. */
  sycldnn::DataFormat getFormat() const { return format_; }

  /** defaults constructor, empty tensor with NCHW format */
  TensorDescriptor()
      : nDims_(0), dims_{}, stride_{}, format_(sycldnn::DataFormat::NCHW) {}

  /** sets the tensor as a 4D tensor.
   * \param format The data format.
   * \param n Batch size.
   * \param c Number of channels.
   * \param h Height.
   * \param w Width.
   * \return A sycldnn::StatusCode showing wheter there were some invalid
   * parameters, or sycldnn::StatusCode::OK
   */
  sycldnn::StatusCode set4d(sycldnn::DataFormat format, int n, int c, int h,
                            int w) {
    SNN_VALIDATE_PARAM(n > 0,
                       "Non strictly positive dimensions are not supported.");
    SNN_VALIDATE_PARAM(c > 0,
                       "Non strictly positive dimensions are not supported.");
    SNN_VALIDATE_PARAM(h > 0,
                       "Non strictly positive dimensions are not supported.");
    SNN_VALIDATE_PARAM(w > 0,
                       "Non strictly positive dimensions are not supported.");
    nDims_ = 4;
    dims_ = {n, c, h, w};
    format_ = format;
    if (format == sycldnn::DataFormat::NCHW)
      stride_ = {c * h * w, h * w, w, 1};
    else if (format == sycldnn::DataFormat::NHWC)
      stride_ = {c * h * w, 1, w * c, c};
    else
      return sycldnn::StatusCode::InvalidParameter;

    return sycldnn::StatusCode::OK;
  }
};

}  // namespace compat

}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_COMPAT_UTILS_HPP

#ifndef SYCLDNN_INCLUDE_COMPAT_UTILS_HPP
#define SYCLDNN_INCLUDE_COMPAT_UTILS_HPP
#include <sycldnn/backend/snn_usm_backend.h>
#include <sycldnn/data_format.h>
#include <sycldnn/filter_format.h>
#include <sycldnn/status.h>
#include <memory>
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
  std::unique_ptr<backend::SNNUSMBackend> b_;

  SNNHandle(const sycl::device_selector& d, const sycl::property_list& props)
      : b_(std::make_unique<backend::SNNUSMBackend>(sycl::queue(d, props))) {}

 public:
  SNNHandle() : b_(nullptr) {}

  /**
   * Constructs an SNNHandle object.
   *
   * @param selector Device to bind the handle to
   * @param props Properties of the queue
   * @return sycldnn::StatusCode::OK if the handle was initialised successfully.
   *         sycldnn::StatusCode::InvalidParameter if the
   *         sycl::property::queue::in_order wasn't specified.
   */
  sycldnn::StatusCode init(const sycl::device_selector& selector,
                           const sycl::property_list& props) {
    if (!props.has_property<sycl::property::queue::in_order>()) {
      return sycldnn::StatusCode::InvalidParameter;
    }

    *this = SNNHandle(selector, props);

    return sycldnn::StatusCode::OK;
  }

  /** \return the SNNBackend */
  backend::SNNUSMBackend& getBackend() { return *b_; }

  /** \return the sycl::queue */
  sycl::queue getQueue() { return b_->get_queue(); }

  /**
   * Set the queue to be used by the backend.
   *
   * @param queue Queue to use. It is expected to have the same context as the
   *              current queue and it must have the
   *              sycl::property::queue::in_order property
   *
   * @return sycldnn::StatusCode::OK if queue was set successfully.
   *          sycldnn::StatusCode::InvalidParameter if the contexts of the
   *          previous queue and the new queue don't match or the new queue is
   *          not an in order queue.
   */
  sycldnn::StatusCode setQueue(sycl::queue queue) {
    if (getQueue().get_context() != queue.get_context()) {
      return sycldnn::StatusCode::InvalidParameter;
    }

    if (!queue.is_in_order()) {
      return sycldnn::StatusCode::InvalidParameter;
    }

    b_.reset(new backend::SNNUSMBackend(queue));

    return sycldnn::StatusCode::OK;
  }
};

/**
 * Creates the SYCL-DNN library context
 *
 * @param handle SNNHandle object to be initialised
 * @param selector Device the context will be bound to
 * @param props SYCL queue properties to use. Note that
 *              sycl::property::queue::in_order is required.
 * @return sycldnn::StatusCode::OK if the creation was successful
 *         sycldnn::StatusCode::InvalidParameter if the
 *         sycl::property::queue::in_order wasn't specified.
 */
sycldnn::StatusCode SNNCreate(
    SNNHandle& handle,
    const sycl::device_selector& selector = sycl::default_selector(),
    const sycl::property_list& props = {sycl::property::queue::in_order()}) {
  return handle.init(selector, props);
}

/**
 * Set the queue to be used by the SNNHandle
 *
 * @param handle Handle for which the queue will be set
 * @param queue Queue to set
 * @return sycldnn::StatusCode::OK if the operation succeeded.
 *         sycldnn::StatusCode::InvalidParameter if the contexts of the previous
 *         and the new queue didn't match or the new queue is an out of order
 *         queue
 */
sycldnn::StatusCode queueSet(SNNHandle& handle, sycl::queue queue) {
  return handle.setQueue(queue);
}

/**
 * class that describes the dimensions, strides, and data format for a
 * tensor. Currently only 4D tensors are supported, with the NCHW or NHWC
 * formats.
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

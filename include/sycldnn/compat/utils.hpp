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

#ifndef SYCLDNN_INCLUDE_COMPAT_UTILS_HPP
#define SYCLDNN_INCLUDE_COMPAT_UTILS_HPP

#include "sycldnn/backend/snn_usm_backend.h"
#include "sycldnn/conv2d/algorithm.h"
#include "sycldnn/data_format.h"
#include "sycldnn/filter_format.h"
#include "sycldnn/status.h"

#include <exception>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

/**
 * \file
 * Contains descriptor and helper classes used in the rest of the compat API.
 */

/** Assertion to provide an error and terminate execution for the compat wrapper
 */
#define SNN_COMPAT_ASSERT(condition, message) \
  do {                                        \
    if (!(condition)) {                       \
      std::cerr << message << std::endl;      \
      std::terminate();                       \
    }                                         \
  } while (0)

namespace sycldnn {
namespace compat {

/** The data type of tensor */
enum class SNNDataType { SNN_FLOAT = 0, SNN_DOUBLE, SNN_HALF };

/** Struct containing performance results */
struct convolutionFwdAlgoPerf_t {
  /** \param algo Vector of selected convolution algorithms. */
  std::vector<conv2d::Algorithm> algo;
  /** \param status Vector of status of convolution algorithms.*/
  std::vector<SNNStatus> status;
  /** \param time Vector of performance timing of convolution algorithm.*/
  std::vector<float> time;
  /** \param memory Vector of workspace size required (defaults to 0).*/
  std::vector<size_t> memory;
};

/**
 * Wrapper around the sycl-dnn backends.
 */
class SNNHandleImpl {
  std::unique_ptr<backend::SNNUSMBackend> b_;

 public:
  SNNHandleImpl() : b_(nullptr) {}
  SNNHandleImpl(std::nullptr_t) : b_(nullptr) {}
  SNNHandleImpl(const sycl::device_selector& d,
                const sycl::property_list& props)
      : b_(std::make_unique<backend::SNNUSMBackend>(sycl::queue(d, props))) {}

  SNNHandleImpl(const SNNHandleImpl& rhs) {
    this->b_.reset(new backend::SNNUSMBackend(*rhs.b_));
  }

  const SNNHandleImpl& operator=(const SNNHandleImpl& rhs) {
    // this.b_ = std::unique_ptr<backend::SNNUSMBackend>(*rhs.b_);
    this->b_.reset(new backend::SNNUSMBackend(*rhs.b_));

    return *this;
  }

  void destroy() { b_.release(); }

  /** \return the SNNBackend */
  backend::SNNUSMBackend* getBackend() { return b_.get(); }

  /** \return the sycl::queue */
  sycl::queue getQueue() { return b_->get_queue(); }


};

class SNNHandle {

 public:
  SNNHandle() : handle_(nullptr) {}
  SNNHandle(std::nullptr_t) : handle_(nullptr) {}
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

    handle_ = new SNNHandleImpl(selector, props);

    return sycldnn::StatusCode::OK;
  }

  /** \return the SNNBackend */
  backend::SNNUSMBackend* getBackend() { return handle_->getBackend(); }

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
    if (handle_->getQueue().get_context() != queue.get_context()) {
      return sycldnn::StatusCode::InvalidParameter;
    }

    if (!queue.is_in_order()) {
      return sycldnn::StatusCode::InvalidParameter;
    }
    auto backend = handle_->getBackend();
    delete backend;

    backend = new backend::SNNUSMBackend(queue);

    return sycldnn::StatusCode::OK;
  }

  SNNHandleImpl* getHandleImpl() { return handle_; }

  sycl::queue getQueue() { return handle_->getQueue(); }

  void destroy() { delete handle_; }

    operator bool() const {
    if (handle_) {
      return true;
    } else {
      return false;
    }
  }

 private:
  SNNHandleImpl* handle_;
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
static void SNNCreate(SNNHandle* handle) {
  if (!handle) {
    return;
  }
  handle->init({sycl::default_selector()}, {sycl::property::queue::in_order()});
}
/*
static void SNNCreate(SNNHandle* handle) {
  if (!handle) {
    return;
  }
  handle->init({sycl::default_selector()}, {sycl::property::queue::in_order()});
}
*/

static void SNNDestroy(SNNHandle handle) { handle.destroy(); }

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
static sycldnn::StatusCode queueSet(SNNHandle* handle, sycl::queue queue) {
  return handle->setQueue(queue);
}

/**
 * Base class for abstracting common features of the Tensor and Filter
 * descriptor classes
 */
class DescriptorBase {
 protected:
  /** Index type for dimensions */
  using Index_t = int;
  /** Number of tensor dimensions (default to 4)*/
  size_t nDims_;
  /** vector containing dimensions of tensor */
  std::vector<Index_t> dims_;

 public:
  /** defaults constructor, empty tensor with format set from derived classes */
  DescriptorBase() : nDims_(4), dims_(4, 0) {}

  /** Constructor which takes number of dimensions descriptor with format set
   * from derived classes
   * \param nDims   Number of dimensions to set for the descriptor (supports 1
   * to 4-D)
   */
  DescriptorBase(int nDims) : nDims_(nDims), dims_(nDims, 0) {
    SNN_COMPAT_ASSERT(nDims > 0 || nDims <= 4,
                      "Unsupported number of dimensions requested!");
  }

  /** defaults destructor */
  virtual ~DescriptorBase() {}

  /** sets the tensor as a 4D tensor.
   * \param format The data format.
   * \param dim0   Value to set in dimension 1
   * \param dim1   Value to set in dimension 2
   * \param dim2   Value to set in dimension 3
   * \param dim3   Value to set in dimension 4
   * \return       A sycldnn::StatusCode showing wheter there were some
   * invalid parameters, or sycldnn::StatusCode::OK
   */
  virtual sycldnn::StatusCode set4d(sycldnn::DataFormat format, int dim0,
                                    int dim1, int dim2, int dim3) = 0;
};

/**
 * class that describes the dimensions, strides, and data format for a
 * tensor. Currently only 4D tensors are supported, with the NCHW or NHWC
 * formats.
 */
class TensorDescriptor : public DescriptorBase {
  /** Vector containing size of stride of tensor for each dimension */
  std::vector<int> stride_;
  /** Data format of tensor defaults to NCHW*/
  sycldnn::DataFormat format_;

 public:
  /** \return data format. */
  sycldnn::DataFormat getFormat() const { return format_; }

  /** \return total size of the tensor (number of elements)*/
  size_t getSize() const {
    return std::accumulate(dims_.begin(), dims_.end(), 1,
                           std::multiplies<int>());
  }

  /** defaults constructor, empty tensor with NCHW format */
  TensorDescriptor()
      : DescriptorBase(), stride_(4, 1), format_(sycldnn::DataFormat::NCHW) {
    SNN_COMPAT_ASSERT(nDims_ == 4,
                      "Cannot call method on non 4-D tensor desc.");
  }

  /** \copydoc DescriptorBase::set4d */
  sycldnn::StatusCode set4d(sycldnn::DataFormat format, int dim0, int dim1,
                            int dim2, int dim3) override final {
    SNN_VALIDATE_PARAM(dim0 > 0 && dim1 > 0 && dim2 > 0 && dim3 > 0,
                       "Non strictly positive dimensions are not supported.");
    nDims_ = 4;
    format_ = format;
    if (format == sycldnn::DataFormat::NCHW) {
      stride_ = {dim1 * dim2 * dim3, dim2 * dim3, dim3, 1};
      dims_ = {dim0, dim1, dim2, dim3};

    } else if (format == sycldnn::DataFormat::NHWC) {
      stride_ = {dim1 * dim2 * dim3, 1, dim3 * dim1, dim1};
      dims_ = {dim0, dim2, dim3, dim1};

    } else
      return sycldnn::StatusCode::InvalidParameter;

    return sycldnn::StatusCode::OK;
  }

  /** This function queries the NCHW params of the previously initialized
   * descriptor object.
   * \param n   Output number of images.
   * \param c   Output number of feature maps per image.
   * \param h   Output height of each feature map.
   * \param w   Output width of each feature map.
   * \return    sycldnn::StatusCode::OK or sycldnn::StatusCode::InvalidParameter
   */
  sycldnn::StatusCode get4dDescriptorDims(int* n, int* c, int* h,
                                          int* w) const {
    SNN_VALIDATE_PARAM(n != nullptr, "Output pointer cannot be null");
    SNN_VALIDATE_PARAM(c != nullptr, "Output pointer cannot be null");
    SNN_VALIDATE_PARAM(h != nullptr, "Output pointer cannot be null");
    SNN_VALIDATE_PARAM(w != nullptr, "Output pointer cannot be null");

    *n = dims_[0];
    if (format_ == sycldnn::DataFormat::NCHW) {
      *c = dims_[1];
      *h = dims_[2];
      *w = dims_[3];
    } else if (format_ == sycldnn::DataFormat::NHWC) {
      *h = dims_[1];
      *w = dims_[2];
      *c = dims_[3];

    } else {
      return sycldnn::StatusCode::InvalidParameter;
    }
    return sycldnn::StatusCode::OK;
  }

  /** This function queries the parameters of the previously initialized
   * descriptor object.
   * \param nStride   Output. Stride between two consecutive images.
   * \param cStride   Output. Stride between two consecutive feature maps.
   * \param hStride   Output. Stride between two consecutive rows.
   * \param wStride   Output. Stride between two consecutive columns.
   * \return          sycldnn::StatusCode::OK or
   *                  sycldnn::StatusCode::InvalidParameter
   */
  sycldnn::StatusCode get4dDescriptorStride(int* nStride, int* cStride,
                                            int* hStride, int* wStride) const {
    SNN_VALIDATE_PARAM(nStride != nullptr, "Output pointer cannot be null");
    SNN_VALIDATE_PARAM(cStride != nullptr, "Output pointer cannot be null");
    SNN_VALIDATE_PARAM(hStride != nullptr, "Output pointer cannot be null");
    SNN_VALIDATE_PARAM(wStride != nullptr, "Output pointer cannot be null");

    *nStride = stride_[0];
    if (format_ == sycldnn::DataFormat::NCHW) {
      *cStride = stride_[1];
      *hStride = stride_[2];
      *wStride = stride_[3];
    } else if (format_ == sycldnn::DataFormat::NHWC) {
      *hStride = stride_[1];
      *wStride = stride_[2];
      *cStride = stride_[3];
    } else {
      return sycldnn::StatusCode::InvalidParameter;
    }
    return sycldnn::StatusCode::OK;
  }

  /** This function queries the parameters of the previously initialized
   * descriptor object.
   * \param dataType  Output data type.
   * \param n         Output number of images.
   * \param c         Output number of feature maps per image.
   * \param h         Output height of each feature map.
   * \param w         Output width of each feature map.
   * \param nStride   Output stride between two consecutive images.
   * \param cStride   Output stride between two consecutive feature maps.
   * \param hStride   Output stride between two consecutive rows.
   * \param wStride   Output stride between two consecutive columns.
   * \return          sycldnn::StatusCode::OK or
   *                  sycldnn::StatusCode::InvalidParameter
   */
  sycldnn::StatusCode getTensor4dDescriptor(SNNDataType* dataType, int* n,
                                            int* c, int* h, int* w,
                                            int* nStride, int* cStride,
                                            int* hStride, int* wStride) const {
    SNN_VALIDATE_PARAM(dataType != nullptr, "Output pointer cannot be null");
    *dataType = SNNDataType::SNN_FLOAT;
    auto returnStatus = get4dDescriptorDims(n, c, h, w);
    if (returnStatus != sycldnn::StatusCode::OK) {
      return returnStatus;
    }
    return get4dDescriptorStride(nStride, cStride, hStride, wStride);
  }
};

/** This function queries the parameters of the previously initialized
 * descriptor object.
 * \param tensorDesc  Input a previously initialized tensor descriptor.
 * \param dataType    Output data type.
 * \param n           Output number of images.
 * \param c           Output number of feature maps per image.
 * \param h           Output height of each feature map.
 * \param w           Output width of each feature map.
 * \param nStride     Output stride between two consecutive images.
 * \param cStride     Output stride between two consecutive feature maps.
 * \param hStride     Output stride between two consecutive rows.
 * \param wStride     Output stride between two consecutive columns.
 * \return            sycldnn::StatusCode::OK or
 *                    sycldnn::StatusCode::InvalidParameter
 */
static sycldnn::StatusCode getTensor4dDescriptor(TensorDescriptor tensorDesc,
                                                 SNNDataType* dataType, int* n,
                                                 int* c, int* h, int* w,
                                                 int* nStride, int* cStride,
                                                 int* hStride, int* wStride) {
  return tensorDesc.getTensor4dDescriptor(dataType, n, c, h, w, nStride,
                                          cStride, hStride, wStride);
}

/** This function sets the parameters of the previously initialized
 * descriptor object.
 * \param tensorDesc  Input a previously initialized tensor descriptor.
 * \param format      Format of the tensor descriptor KCHW or KHWC
 * \param dataType    Output data type.
 * \param n           Output number of images.
 * \param c           Output number of feature maps per image.
 * \param h           Output height of each feature map.
 * \param w           Output width of each feature map.
 * \return            sycldnn::StatusCode::OK or
 *                    sycldnn::StatusCode::InvalidParameter
 */
static sycldnn::StatusCode setTensor4dDescriptor(TensorDescriptor& tensorDesc,
                                                 sycldnn::DataFormat format,
                                                 SNNDataType dataType, int n,
                                                 int c, int h, int w) {
  SNN_UNUSED_VAR(dataType);
  return tensorDesc.set4d(format, n, c, h, w);
}

/** This function sets the parameters of the previously initialized
 * descriptor object.
 * \param tensorDesc  Input a previously initialized tensor descriptor.
 * \param format      Format of the tensor descriptor KCHW or KHWC
 * \param dataType    Output data type.
 * \param n           Output number of images.
 * \param c           Output number of feature maps per image.
 * \param h           Output height of each feature map.
 * \param w           Output width of each feature map.
 * \return            sycldnn::StatusCode::OK or
 *                    sycldnn::StatusCode::InvalidParameter
 */
static sycldnn::StatusCode setTensor4dDescriptor(TensorDescriptor& tensorDesc,
                                                 sycldnn::DataFormat format,
                                                 SNNDataType dataType, int n,
                                                 int c, int h, int w, int s_n,
                                                 int s_c, int s_h, int s_w) {
  SNN_UNUSED_VAR(dataType);
  return tensorDesc.set4d(format, n, c, h, w);
}

/** TODO ADD
 *
 */
static sycldnn::StatusCode createTensorDescriptor(
    TensorDescriptor* tensorDesc) {
  if (!tensorDesc) {
    tensorDesc = new TensorDescriptor;
    return sycldnn::StatusCode::OK;
  } else {
    return sycldnn::StatusCode::InvalidParameter;
  }
}

/** TODO
 *
 */
static sycldnn::StatusCode destroyTensorDescriptor(
    TensorDescriptor* tensorDesc) {
  if (tensorDesc) {
    delete tensorDesc;
    return sycldnn::StatusCode::OK;
  }
  return sycldnn::StatusCode::InvalidParameter;
}

}  // namespace compat

}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_COMPAT_UTILS_HPP

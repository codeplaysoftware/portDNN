#ifndef DNNCOMPAT_ACTIVATION_H
#define DNNCOMPAT_ACTIVATION_H

#include "nan.hpp"
#include "scaling.hpp"
#include "types.hpp"
#include "utils.hpp"

#include "sycldnn/pointwise/direction.h"
#include "sycldnn/pointwise/launch.h"
#include "sycldnn/pointwise/operators.h"
#include "sycldnn/pointwise/params.h"

namespace sycldnn {
namespace compat {
/** Activation mode */
enum class ActivationMode {
  UNDEFINED = 0,
  ACTIVATION_SIGMOID,
  ACTIVATION_RELU,
  ACTIVATION_LOG
};

/**
 * class containing the activation mode, nan propagation choice and coefficient
 * for the activation operation. operation.
 * */
class ActivationDescriptor {
  ActivationMode mode_;
  NanPropagation reluNanOpt_;
  double coef_ = 0.f;

 public:
  ActivationDescriptor()
      : mode_{ActivationMode::ACTIVATION_LOG},
        reluNanOpt_{NanPropagation::NOT_PROPAGATE_NAN} {}

  /** \return coefficient*/
  double getCoef() const { return coef_; }

  /** \return activation mode*/
  ActivationMode getActivationMode() const { return mode_; };

  /**
   * Internal API for setting the activation descriptor
   * \param mode The activation operator to be used
   * \param reluNanOpt The NaN propagation option to use
   * \param coef The coefficient used in the operation
   * \return sycldnn::StatusCode
   */
  sycldnn::StatusCode set(
      ActivationMode mode = ActivationMode::ACTIVATION_RELU,
      NanPropagation reluNanOpt = NanPropagation::NOT_PROPAGATE_NAN,
      double coef = 0.f) {
    SNN_VALIDATE_PARAM(mode == ActivationMode::ACTIVATION_RELU ||
                           mode == ActivationMode::ACTIVATION_LOG,
                       "Only relu or log is currently supported");
    mode_ = mode;
    reluNanOpt_ = reluNanOpt;
    coef_ = coef;
    return StatusCode::OK;
  }
};

/**
 * Sets the descriptor for activation operation
 * \param mode The activation operator to be used
 * \param reluNanOpt The NaN propagation option to use
 * \param coef The coefficient used in the operation
 * \return sycldnn::StatusCode
 */
sycldnn::StatusCode setActivationDescriptor(
    ActivationDescriptor& activationDesc, ActivationMode mode,
    NanPropagation reluNanOpt, double coef) {
  return activationDesc.set(mode, reluNanOpt, coef);
}

/**
 * Performs the activation forward operation.
 * \tparam T Datatype of input/output pointers
 * \param handle The SNNHandle.
 * \param activationDesc Descriptor for the activation operation.
 * \param alpha Scaling factor used to blend y output.
 * \param xDesc Descriptor for the input tensor.
 * \param x Pointer to device memory for the input tensor.
 * \param beta Scaling factor used to blend y output.
 * \param yDesc Descriptor for the output tensor.
 * \param y Pointer to device memory for the output.
 * \return SNNStatus for the operation.
 */
template <typename T>
SNNStatus activationForward(SNNHandle& handle,
                            ActivationDescriptor const& activationDesc,
                            T const* alpha, TensorDescriptor const& xDesc,
                            T const* x, const T* beta,
                            TensorDescriptor const& yDesc, T* y) {
  ScalingParams scParams(handle.getBackend(), alpha, beta, yDesc.getSize(), y,
                         true);
  auto event = scParams.constructMem(handle.getBackend());
  if (!scParams.isAlphaZero()) {
    auto const size = xDesc.getSize();
    sycldnn::SNNStatus status;
    if (activationDesc.getActivationMode() == ActivationMode::ACTIVATION_RELU) {
      status = sycldnn::pointwise::launch<T, sycldnn::pointwise::Relu,
                                          sycldnn::pointwise::Forward>(
          x, y, size, handle.getBackend());
    } else if (activationDesc.getActivationMode() ==
               ActivationMode::ACTIVATION_LOG) {
      status = sycldnn::pointwise::launch<T, sycldnn::pointwise::Log,
                                          sycldnn::pointwise::Forward>(
          x, y, size, handle.getBackend());
    } else {
      return {StatusCode::InvalidParameter};
    }

    if (status.status != StatusCode::OK) {
      return status;
    }
    event = status.event;
  }
  return scParams.applyScaling(handle.getBackend(), {event});
}

}  // namespace compat
}  // namespace sycldnn
#endif  // DNNCOMPAT_ACTIVATION_H

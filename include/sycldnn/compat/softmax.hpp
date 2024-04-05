#ifndef COMPAT_SOFTMAX_HPP
#define COMPAT_SOFTMAX_HPP

#include "sycldnn/softmax/direction.h"
#include "sycldnn/softmax/launch.h"
#include "sycldnn/softmax/params.h"

#include "scaling.hpp"
#include "utils.hpp"

namespace sycldnn {
namespace compat {

/** Softmax mode */
enum class SoftmaxMode {
  UNDEFINED = 0,
  SOFTMAX_MODE_INSTANCE,
  SOFTMAX_MODE_CHANNEL
};

/** Softmax algorithm */
enum class SoftmaxAlgorithm {
  UNDEFINED = 0,
  SOFTMAX_FAST,
  SOFTMAX_ACCURATE,
  SOFTMAX_LOG
};

/**
 * Performs the softmax forward operation.
 * \tparam T Datatype of input/output pointers
 * \param handle The SNNHandle.
 * \param algorithm Softmax algorithm.
 * \param mode Softmax mode.
 * \param alpha Scaling factor used to blend y output.
 * \param xDesc Descriptor for the input tensor.
 * \param x Pointer to device memory for the input tensor.
 * \param beta Scaling factor used to blend y output.
 * \param yDesc Descriptor for the output tensor.
 * \param y Pointer to device memory for the output.
 * \return SNNStatus for the operation.
 */
template <typename T>
SNNStatus softmaxForward(SNNHandle& handle, SoftmaxAlgorithm algorithm,
                         SoftmaxMode mode, const T* alpha,
                         const TensorDescriptor& xDesc, const T* x,
                         const T* beta, const TensorDescriptor& yDesc, T* y) {
  SNN_VALIDATE_PARAM(
      algorithm == SoftmaxAlgorithm::SOFTMAX_ACCURATE,
      "Currently SOFTMAX_ACCURATE is the only supported algorithm.");
  SNN_VALIDATE_PARAM(mode == SoftmaxMode::SOFTMAX_MODE_CHANNEL,
                     "Currently, channel mode is only supported.");

  int n, c, h, w;
  xDesc.get4dDescriptorDims(&n, &c, &h, &w);

  sycldnn::softmax::SoftmaxParams params = {c, n, h, w};

  auto& backend = handle.getBackend();
  auto workspace =
      backend.allocate<T>(params.batch * params.rows * params.cols);

  ScalingParams scParams(backend, alpha, beta, yDesc.getSize(), y, true);
  auto event = scParams.constructMem(backend);
  if (!scParams.isAlphaZero()) {
    auto status = sycldnn::softmax::launch<T, sycldnn::softmax::Forward,
                                           sycldnn::backend::SNNUSMBackend>(
        x, workspace, y, params, backend, {event});

    if (status.status != sycldnn::StatusCode::OK) {
      return status;
    }
    event = status.event;
  }
  return scParams.applyScaling(backend, {event});
}

}  // namespace compat
}  // namespace sycldnn
#endif  // COMPAT_SOFTMAX_HPP

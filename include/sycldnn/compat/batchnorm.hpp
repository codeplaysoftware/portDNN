#ifndef SYCLDNN_INCLUDE_COMPAT_BATCHNORM_HPP
#define SYCLDNN_INCLUDE_COMPAT_BATCHNORM_HPP
#include "sycldnn/batchnorm/launch.h"
#include "sycldnn/batchnorm/params.h"
#include "utils.hpp"

#include "sycldnn/helpers/event_handling.h"

/**
 * \file
 * Wrapper API for batch normalization.
 */

namespace sycldnn {
namespace compat {

/** Defines types of batch normalization. */
enum BatchNormMode {
  BATCHNORM_PER_ACTIVATION,  // bias and scale are 1xCxHxW
  BATCHNORM_SPATIAL,         // bias and scale are 1xCx1x1
  BATCHNORM_SPATIAL_PERSISTENT
};

namespace internal {
/**
 * Validates descriptors passed to batchnorm operations.
 * \param xDesc Input descriptor.
 * \param yDesc Output descriptor.
 * \param bnScaleBiasMeanVarDesc Scale/Bias/Mean/Variance descriptor.
 * \param mode Batch normalization mode.
 * \return Validation status.
 */
SNNStatus validateBatchnormParams(const TensorDescriptor xDesc,
                                  const TensorDescriptor yDesc,
                                  const TensorDescriptor bnScaleBiasMeanVarDesc,
                                  BatchNormMode mode) {
  SNN_VALIDATE_PARAM(mode != BatchNormMode::BATCHNORM_PER_ACTIVATION,
                     "PER_ACTIVATION batchnorm is currently unsupported");
  int xN, xC, xH, xW;
  int yN, yC, yH, yW;

  xDesc.get4dDescriptorDims(&xN, &xC, &xH, &xW);
  yDesc.get4dDescriptorDims(&yN, &yC, &yH, &yW);
  SNN_VALIDATE_PARAM(xN == yN, "Input and output N mismatch");
  SNN_VALIDATE_PARAM(xC == yC, "Input and output C mismatch");
  SNN_VALIDATE_PARAM(xH == yH, "Input and output H mismatch");
  SNN_VALIDATE_PARAM(xW == yW, "Input and output W mismatch");
  SNN_VALIDATE_PARAM(xDesc.getFormat() == yDesc.getFormat(),
                     "Input and output format mismatch");

  int bnsN, bnsC, bnsH, bnsW;
  bnScaleBiasMeanVarDesc.get4dDescriptorDims(&bnsN, &bnsC, &bnsH, &bnsW);

  if (mode == BatchNormMode::BATCHNORM_PER_ACTIVATION) {
    SNN_VALIDATE_PARAM(bnsN == 1,
                       "Wrong scale/bias desc for PER_ACTIVATION mode");
    SNN_VALIDATE_PARAM(bnsC == xC,
                       "Wrong scale/bias desc for PER_ACTIVATION mode");
    SNN_VALIDATE_PARAM(bnsH == xH,
                       "Wrong scale/bias desc for PER_ACTIVATION mode");
    SNN_VALIDATE_PARAM(bnsW == xW,
                       "Wrong scale/bias desc for PER_ACTIVATION mode");
  } else if (mode == BatchNormMode::BATCHNORM_SPATIAL ||
             mode == BatchNormMode::BATCHNORM_SPATIAL_PERSISTENT) {
    SNN_VALIDATE_PARAM(bnsN == 1, "Wrong scale/bias desc for SPATIAL mode");
    SNN_VALIDATE_PARAM(bnsC == xC, "Wrong scale/bias desc for SPATIAL mode");
    SNN_VALIDATE_PARAM(bnsH == 1, "Wrong scale/bias desc for SPATIAL mode");
    SNN_VALIDATE_PARAM(bnsW == 1, "Wrong scale/bias desc for SPATIAL mode");
  } else {
    return StatusCode::InvalidParameter;
  }

  return StatusCode::OK;
}

/**
 * Converts the descriptor into a sycldnn::batchnorm::BatchNormParams
 * \param xDesc        Input descriptor.
 * \param is_training  True if performing batchnorm in training mode.
 * \param epsilon      Epsilon value to avoid division by zero.
 * \return             Converted sycldnn::BatchNormParams.
 */
batchnorm::BatchNormParams descToSnnBatchnormParams(
    const TensorDescriptor& xDesc, bool is_training, float epsilon) {
  batchnorm::BatchNormParams params{};
  int n, c, h, w;
  xDesc.get4dDescriptorDims(&n, &c, &h, &w);

  params.batch = n;
  params.cols = w;
  params.rows = h;
  params.channels = c;
  params.is_training = is_training;
  params.epsilon = epsilon;
  params.input_format = xDesc.getFormat();

  return params;
}
}  // namespace internal

/**
 * Executes forward pass during inference.
 * \param handle The SNNHandle.
 * \param mode Batch normalization mode.
 * \param alpha Scaling parameter, currently unused.
 * \param beta Scaling parameter, currently unused.
 * \param xDesc Input tensor descriptor.
 * \param x Pointer to input device memory.
 * \param yDesc Output tensor descriptor.
 * \param y Pointer to output device memory.
 * \param bnScaleBiasMeanVarDesc Scale/Bias/Mean/Variance descriptor.
 * \param bnScale Pointer to device memory for the scale parameter.
 * \param bnBias Pointer to device memory for the bias parameter.
 * \param estimatedMean Pointer to device memory for the estimated mean.
 * \param estimatedVariance Pointer to device memory for the estimated variance.
 * \param epsilon Epsilon value to avoid division by zero.
 * \return Status of the operation.
 */
template <typename ValueT = float>
SNNStatus batchNormalizationForwardInference(
    SNNHandle& handle, BatchNormMode mode, const void* alpha, const void* beta,
    const TensorDescriptor& xDesc, const void* x, const TensorDescriptor& yDesc,
    void* y, const TensorDescriptor& bnScaleBiasMeanVarDesc,
    const void* bnScale, const void* bnBias, const void* estimatedMean,
    const void* estimatedVariance, double epsilon) {
  SNNStatus validationStatus = internal::validateBatchnormParams(
      xDesc, yDesc, bnScaleBiasMeanVarDesc, mode);
  if (validationStatus.status != StatusCode::OK) return validationStatus;

  SNN_UNUSED_VAR(alpha);
  SNN_UNUSED_VAR(beta);

  batchnorm::BatchNormParams params =
      internal::descToSnnBatchnormParams(xDesc, false, epsilon);

  return batchnorm::launch<ValueT, sycldnn::backend::SNNUSMBackend,
                           batchnorm::Forward>(
      static_cast<const ValueT*>(x), static_cast<const ValueT*>(bnScale),
      static_cast<const ValueT*>(bnBias),
      static_cast<const ValueT*>(estimatedMean),
      static_cast<const ValueT*>(estimatedVariance), nullptr, nullptr,
      static_cast<ValueT*>(y), params, handle.getBackend());
}

/**
 * Executes forward pass during training.
 * \param handle The SNNHandle.
 * \param mode Batch normalization mode.
 * \param alpha Scaling parameter, currently unused.
 * \param beta Scaling parameter, currently unused.
 * \param xDesc Input tensor descriptor.
 * \param x Pointer to input device memory.
 * \param yDesc Output tensor descriptor.
 * \param y Pointer to output device memory.
 * \param bnScaleBiasMeanVarDesc Scale/Bias/Mean/Variance descriptor.
 * \param bnScale Pointer to device memory for the scale parameter.
 * \param bnBias Pointer to device memory for the bias parameter.
 * \param exponentialAverageFactor Moving average coefficient.
 * \param resultRunningMean Pointer to device memory for the output running
 * mean.
 * \param resultRunningVariance Pointer to device memory for the output
 * running variance.
 * \param epsilon Epsilon value to avoid division by zero.
 * \param resultSaveMean Optional cache for saving the running mean.
 * \param resultSaveInvVariance Optional cache for saving running variance.
 * \return Status of the operation.
 */
template <typename ValueT = float>
SNNStatus batchNormalizationForwardTraining(
    SNNHandle& handle, BatchNormMode mode, const void* alpha, const void* beta,
    const TensorDescriptor xDesc, const void* x, const TensorDescriptor yDesc,
    void* y, const TensorDescriptor bnScaleBiasMeanVarDesc, const void* bnScale,
    const void* bnBias, double exponentialAverageFactor,
    void* resultRunningMean, void* resultRunningVariance, double epsilon,
    void* resultSaveMean, void* resultSaveInvVariance) {
  SNN_UNUSED_VAR(alpha);
  SNN_UNUSED_VAR(beta);
  SNNStatus validationStatus = internal::validateBatchnormParams(
      xDesc, yDesc, bnScaleBiasMeanVarDesc, mode);
  if (validationStatus.status != StatusCode::OK) return validationStatus;

  SNN_VALIDATE_PARAM(
      !resultSaveMean == !resultSaveInvVariance,
      "The optional cache pointers need to either be both valid or both null");

  batchnorm::BatchNormParams params =
      internal::descToSnnBatchnormParams(xDesc, true, epsilon);
  params.momentum = 1 - exponentialAverageFactor;

  auto C = params.channels;
  ValueT* out_mean_ptr;
  ValueT* out_var_ptr;

  if (!resultSaveMean) {
    out_mean_ptr = sycl::malloc_device<ValueT>(C, handle.getQueue());
    out_var_ptr = sycl::malloc_device<ValueT>(C, handle.getQueue());
  } else {
    out_mean_ptr = static_cast<ValueT*>(resultSaveMean);
    out_var_ptr = static_cast<ValueT*>(resultSaveInvVariance);
  }

  auto batchnorm_status =
      batchnorm::launch<ValueT, sycldnn::backend::SNNUSMBackend,
                        batchnorm::Forward>(
          static_cast<const ValueT*>(x), static_cast<const ValueT*>(bnScale),
          static_cast<const ValueT*>(bnBias),
          static_cast<ValueT*>(resultRunningMean),
          static_cast<ValueT*>(resultRunningVariance),
          static_cast<ValueT*>(out_mean_ptr), static_cast<ValueT*>(out_var_ptr),
          static_cast<ValueT*>(y), params, handle.getBackend());

  // copy back data to match cuDNN parameters.
  auto q = handle.getQueue();
  size_t C_size = C * sizeof(ValueT);
  std::vector<sycl::event> dep_events;
  auto copy_mean_event = q.memcpy(resultRunningMean, out_mean_ptr, C_size,
                                  {batchnorm_status.event});
  auto copy_var_event = q.memcpy(resultRunningVariance, out_var_ptr, C_size,
                                 {batchnorm_status.event});
  dep_events.push_back(copy_mean_event);
  dep_events.push_back(copy_var_event);

  sycl::event final_event;
  if (!resultSaveMean) {
    final_event = q.submit([&](sycl::handler& cgh) {
      cgh.depends_on(dep_events);
      cgh.host_task([=]() {
        sycl::free(out_mean_ptr, q);
        sycl::free(out_var_ptr, q);
      });
    });
  } else {
    final_event = sycldnn::helpers::multi_event_to_one(dep_events, q);
  }

  return {final_event, sycldnn::StatusCode::OK};
}

/**
 * Executes backward pass.
 * \param handle The SNNHandle.
 * \param mode Batch normalization mode.
 * \param alphaDataDiff Scaling parameter, currently unused.
 * \param betaDataDiff Scaling parameter, currently unused.
 * \param alphaParamDiff Scaling parameter, currently unused.
 * \param betaParamDiff Scaling parameter, currently unused.
 * \param xDesc Input tensor descriptor.
 * \param x Pointer to input device memory.
 * \param dyDesc Backpropagation differential tensor descriptor.
 * \param dy Pointer to backpropagation differential device memory.
 * \param dxDesc Differential output tensor descriptor.
 * \param dx Pointer to differential output device memory.
 * \param bnScaleBiasDiffDesc Scale/Bias differential descriptor.
 * \param bnScale Pointer to device memory for the scale parameter.
 * \param resultBnScaleDiff Pointer to device memory for the scale differential.
 * \param resultBnBiasDiff Pointer to device memory for the bias differential.
 * \param epsilon Epsilon value to avoid division by zero.
 * \param savedMean Optional cache for the running mean, currently unused.
 * \param savedInvVariance Optional cache for the running variance, currently
 * unused.
 * \return Status of the operation.
 */
template <typename ValueT = float>
SNNStatus batchNormalizationBackward(
    SNNHandle& handle, BatchNormMode mode, const void* alphaDataDiff,
    const void* betaDataDiff, const void* alphaParamDiff,
    const void* betaParamDiff, const TensorDescriptor xDesc, const void* x,
    const TensorDescriptor dyDesc, const void* dy,
    const TensorDescriptor dxDesc, void* dx,
    const TensorDescriptor bnScaleBiasDiffDesc, const void* bnScale,
    void* resultBnScaleDiff, void* resultBnBiasDiff, double epsilon,
    const void* savedMean, const void* savedInvVariance) {
  for (const auto& desc : {dxDesc, dyDesc}) {
    SNNStatus validationStatus = internal::validateBatchnormParams(
        xDesc, desc, bnScaleBiasDiffDesc, mode);
    if (validationStatus.status != StatusCode::OK) return validationStatus;
  }

  SNN_UNUSED_VAR(alphaParamDiff);
  SNN_UNUSED_VAR(betaParamDiff);
  SNN_UNUSED_VAR(savedMean);
  SNN_UNUSED_VAR(savedInvVariance);
  SNN_UNUSED_VAR(alphaDataDiff);
  SNN_UNUSED_VAR(betaDataDiff);

  batchnorm::BatchNormParams params =
      internal::descToSnnBatchnormParams(xDesc, true, epsilon);

  return batchnorm::launch<ValueT, sycldnn::backend::SNNUSMBackend,
                           batchnorm::Gradient>(
      static_cast<const ValueT*>(x), static_cast<const ValueT*>(dy),
      static_cast<const ValueT*>(bnScale), nullptr, nullptr,
      static_cast<ValueT*>(resultBnScaleDiff),
      static_cast<ValueT*>(resultBnBiasDiff), static_cast<ValueT*>(dx), params,
      handle.getBackend());
}

}  // namespace compat
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_COMPAT_BATCHNORM_HPP

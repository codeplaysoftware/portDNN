#ifndef PORTDNN_INCLUDE_COMPAT_BATCHNORM_HPP
#define PORTDNN_INCLUDE_COMPAT_BATCHNORM_HPP
#include "portdnn/batchnorm/launch.h"
#include "portdnn/batchnorm/params.h"
#include "scaling.hpp"
#include "utils.hpp"

#include "portdnn/helpers/event_handling.h"

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
 * \param alpha Scaling parameter used to blend y output.
 * \param beta Scaling parameter used to blend y output.
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
template <typename ValueT>
SNNStatus batchNormalizationForwardInference(
    SNNHandle& handle, BatchNormMode mode, const ValueT* alpha,
    const ValueT* beta, const TensorDescriptor& xDesc, const void* x,
    const TensorDescriptor& yDesc, void* y,
    const TensorDescriptor& bnScaleBiasMeanVarDesc, const void* bnScale,
    const void* bnBias, const void* estimatedMean,
    const void* estimatedVariance, double epsilon) {
  SNNStatus validationStatus = internal::validateBatchnormParams(
      xDesc, yDesc, bnScaleBiasMeanVarDesc, mode);
  if (validationStatus.status != StatusCode::OK) return validationStatus;

  ScalingParams scParams(handle.getBackend(), alpha, beta, yDesc.getSize(), y);
  SNNStatus batchnormStatus;
  batchnormStatus.event = scParams.constructMem(handle.getBackend());

  if (!scParams.isAlphaZero()) {
    batchnorm::BatchNormParams batchnormParams =
        internal::descToSnnBatchnormParams(xDesc, false, epsilon);

    batchnormStatus = batchnorm::launch<ValueT, sycldnn::backend::SNNUSMBackend,
                                        batchnorm::Forward>(
        static_cast<const ValueT*>(x), static_cast<const ValueT*>(bnScale),
        static_cast<const ValueT*>(bnBias),
        static_cast<const ValueT*>(estimatedMean),
        static_cast<const ValueT*>(estimatedVariance), nullptr, nullptr,
        static_cast<ValueT*>(y), batchnormParams, handle.getBackend(),
        {batchnormStatus.event});
  }

  return scParams.applyScaling(handle.getBackend(), {batchnormStatus.event});
}

/**
 * Executes forward pass during training.
 * \param handle The SNNHandle.
 * \param mode Batch normalization mode.
 * \param alpha Scaling parameter used to blend y output.
 * \param beta Scaling parameter used to blend y output.
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
template <typename ValueT>
SNNStatus batchNormalizationForwardTraining(
    SNNHandle& handle, BatchNormMode mode, const ValueT* alpha,
    const ValueT* beta, const TensorDescriptor xDesc, const void* x,
    const TensorDescriptor yDesc, void* y,
    const TensorDescriptor bnScaleBiasMeanVarDesc, const void* bnScale,
    const void* bnBias, double exponentialAverageFactor,
    void* resultRunningMean, void* resultRunningVariance, double epsilon,
    void* resultSaveMean, void* resultSaveInvVariance) {
  SNNStatus validationStatus = internal::validateBatchnormParams(
      xDesc, yDesc, bnScaleBiasMeanVarDesc, mode);
  if (validationStatus.status != StatusCode::OK) return validationStatus;

  SNN_VALIDATE_PARAM(
      !resultSaveMean == !resultSaveInvVariance,
      "The optional cache pointers need to either be both valid or both null");

  ScalingParams scParams(handle.getBackend(), alpha, beta, yDesc.getSize(), y,
                         true);
  auto constructMemEvent = scParams.constructMem(handle.getBackend());

  batchnorm::BatchNormParams batchnormParams =
      internal::descToSnnBatchnormParams(xDesc, true, epsilon);
  batchnormParams.momentum = 1 - exponentialAverageFactor;

  auto c = batchnormParams.channels;
  ValueT* outMeanPtr = nullptr;
  ValueT* outVarPtr = nullptr;

  if (!resultSaveMean) {
    outMeanPtr = sycl::malloc_device<ValueT>(c, handle.getQueue());
    outVarPtr = sycl::malloc_device<ValueT>(c, handle.getQueue());
  } else {
    outMeanPtr = static_cast<ValueT*>(resultSaveMean);
    outVarPtr = static_cast<ValueT*>(resultSaveInvVariance);
  }

  SNNStatus batchnormStatus =
      batchnorm::launch<ValueT, sycldnn::backend::SNNUSMBackend,
                        batchnorm::Forward>(
          static_cast<const ValueT*>(x), static_cast<const ValueT*>(bnScale),
          static_cast<const ValueT*>(bnBias),
          static_cast<ValueT*>(resultRunningMean),
          static_cast<ValueT*>(resultRunningVariance),
          static_cast<ValueT*>(outMeanPtr), static_cast<ValueT*>(outVarPtr),
          static_cast<ValueT*>(y), batchnormParams, handle.getBackend(),
          {constructMemEvent});

  // copy back data to match cuDNN parameters.
  auto q = handle.getQueue();
  size_t cSize = c * sizeof(ValueT);
  auto copyMeanEvent =
      q.memcpy(resultRunningMean, outMeanPtr, cSize, {batchnormStatus.event});
  auto copyVarEvent = q.memcpy(resultRunningVariance, outVarPtr, cSize,
                               {batchnormStatus.event});

  std::vector<cl::sycl::event> batchnormEventVector{copyMeanEvent,
                                                    copyVarEvent};

  if (!resultSaveMean) {
    auto e = q.submit([&](sycl::handler& cgh) {
      cgh.depends_on(batchnormEventVector);
      cgh.host_task([=]() {
        sycl::free(outMeanPtr, q);
        sycl::free(outVarPtr, q);
      });
    });
    batchnormEventVector.clear();
    batchnormEventVector.push_back(e);
  }

  return scParams.applyScaling(handle.getBackend(), batchnormEventVector);
}

/**
 * Executes backward pass.
 * \param handle The SNNHandle.
 * \param mode Batch normalization mode.
 * \param alphaDataDiff Scaling parameter used to blend dx output.
 * \param betaDataDiff Scaling parameter used to blend dx output.
 * \param alphaParamDiff Scaling parameter used to blend resultBnScaleDiff
 *                       and resultBnBiasDiff.
 * \param betaParamDiff Scaling parameter used to blend resultBnScaleDiff
 *                      and resultBnBiasDiff.
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
 *                         unused.
 * \return Status of the operation.
 */
template <typename ValueT>
SNNStatus batchNormalizationBackward(
    SNNHandle& handle, BatchNormMode mode, const ValueT* alphaDataDiff,
    const ValueT* betaDataDiff, const ValueT* alphaParamDiff,
    const ValueT* betaParamDiff, const TensorDescriptor xDesc, const void* x,
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

  SNN_UNUSED_VAR(savedMean);
  SNN_UNUSED_VAR(savedInvVariance);

  std::vector<cl::sycl::event> batchnormEventVector;

  ScalingParams scDataDiff(handle.getBackend(), alphaDataDiff, betaDataDiff,
                           dxDesc.getSize(), dx, true);
  auto constructMemEvent = scDataDiff.constructMem(handle.getBackend());
  batchnormEventVector.push_back(constructMemEvent);

  ScalingParams scScaleDiff(handle.getBackend(), alphaParamDiff, betaParamDiff,
                            bnScaleBiasDiffDesc.getSize(), resultBnScaleDiff,
                            true);
  constructMemEvent = scScaleDiff.constructMem(handle.getBackend());
  batchnormEventVector.push_back(constructMemEvent);

  ScalingParams scBiasDiff(handle.getBackend(), alphaParamDiff, betaParamDiff,
                           bnScaleBiasDiffDesc.getSize(), resultBnBiasDiff,
                           true);
  constructMemEvent = scBiasDiff.constructMem(handle.getBackend());
  batchnormEventVector.push_back(constructMemEvent);

  SNNStatus batchnormStatus;

  if (!scDataDiff.isAlphaZero() || !scScaleDiff.isAlphaZero()) {
    batchnorm::BatchNormParams params =
        internal::descToSnnBatchnormParams(xDesc, true, epsilon);

    batchnormStatus = batchnorm::launch<ValueT, sycldnn::backend::SNNUSMBackend,
                                        batchnorm::Gradient>(
        static_cast<const ValueT*>(x), static_cast<const ValueT*>(dy),
        static_cast<const ValueT*>(bnScale), nullptr, nullptr,
        static_cast<ValueT*>(resultBnScaleDiff),
        static_cast<ValueT*>(resultBnBiasDiff), static_cast<ValueT*>(dx),
        params, handle.getBackend(), batchnormEventVector);

    batchnormEventVector.clear();
    batchnormEventVector.push_back(batchnormStatus.event);
  }
  std::vector<cl::sycl::event> batchnormEventVectorFinal;
  batchnormStatus =
      scDataDiff.applyScaling(handle.getBackend(), batchnormEventVector);
  batchnormEventVectorFinal.push_back(batchnormStatus.event);
  batchnormStatus =
      scScaleDiff.applyScaling(handle.getBackend(), batchnormEventVector);
  batchnormEventVectorFinal.push_back(batchnormStatus.event);
  batchnormStatus =
      scBiasDiff.applyScaling(handle.getBackend(), batchnormEventVector);
  batchnormEventVectorFinal.push_back(batchnormStatus.event);

  auto q = handle.getQueue();
  cl::sycl::event syclEvent =
      sycldnn::helpers::multi_event_to_one(batchnormEventVectorFinal, q);
  return {syclEvent, sycldnn::StatusCode::OK};
}
}  // namespace compat
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_COMPAT_BATCHNORM_HPP

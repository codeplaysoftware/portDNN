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

#ifndef PORTDNN_INCLUDE_COMPAT_SCALING_HPP
#define PORTDNN_INCLUDE_COMPAT_SCALING_HPP

#include "portdnn/binaryop/launch.h"
#include "portdnn/binaryop/operators.h"
#include "portdnn/binaryop/params.h"

#include "portdnn/helpers/event_handling.h"
#include "portdnn/helpers/mem_utils.h"

/**
 * \file
 * Contains the logic to support scaling parameters.
 */

namespace sycldnn {
namespace compat {
/**
 * The implementation of scaling parameters for the supported operators
 */
template <typename ValueT, typename Backend = backend::SNNUSMBackend>
class ScalingParams {
 public:
  /** Output device memory pointer*/
  void* y;
  /** Temporary variable to save previous y result*/
  ValueT* yTmp = nullptr;
  /** Scaling parameter applied to the current device output pointer */
  const ValueT* alpha;
  /** Scaling parameter applied to the previous device output pointer */
  const ValueT* beta;
  /** Number of total elements in the device output pointer */
  unsigned int ySize;
  /** Scaling alpha parameter in the device */
  ValueT* devAlpha = nullptr;
  /** Scaling beta parameter in the device */
  ValueT* devBeta = nullptr;
  /** Flag to launch the multiplication and addition call */
  bool enableMulAdd = false;
  /** Flag to determine if current object is used in
   * batchNormalizationForwardTraining */
  bool isBatchnormFwdTr = false;
  /** Queue being used by the ScalingParams class */
  sycl::queue q;

  /** This constructor sets the parameters we need to use inside
   * ScalingParams class
   * \param backend             The backend being used by the Operator
   * \param alpha               Alpha scaling parameter to scale current output
   * \param beta                Beta scaling parameter to scale previous output
   * \param ySize               Size of Operator ouput
   * \param y                   Device pointer to the Operator output
   * \param isBatchnormFwdTr    Boolean flag to check batch normalization
   * forward training call
   */
  ScalingParams(Backend& backend, const ValueT* alpha, const ValueT* beta,
                unsigned int ySize, void* y, bool isBatchnormFwdTr = false) {
    this->y = y;
    this->alpha = alpha;
    this->beta = beta;
    this->ySize = ySize;
    this->q = backend.get_queue();
    this->isBatchnormFwdTr = isBatchnormFwdTr;
  }

  /**
   * Performs floating point comparison
   * \param a         Floating point number to compare
   * \param b         Floating point number to compare
   * \return          boolean true if values are almost equal
   */
  bool isSame(ValueT a, ValueT b) { return (abs(a - b) < 1e-9); }

  /**
   * Checks if alpha parameter is equal to zero.
   * \return    true if alpha parameter is zero, false
   *            in any other case.
   */
  bool isAlphaZero() { return isSame(*this->alpha, 0.f); }

  /**
   * Checks if beta parameter is equal to zero.
   * \return    true if beta parameter is zero, false
   *            in any other case.
   */
  bool isBetaZero() { return isSame(*this->beta, 0.f); }

  /**
   * Checks if alpha parameter is equal to one.
   * \return    true if alpha parameter is one, false
   *            in any other case
   */
  bool isAlphaOne() { return isSame(*this->alpha, 1.f); }

  /**
   * Checks if beta parameter is equal to one.
   * \return    true if beta parameter is one false
   *            in any other case
   */
  bool isBetaOne() { return isSame(*this->beta, 1.f); }

  /**
   * Performs the multiply addition step with the scaling parameters
   * \param params              BinaryParams to perform the internal binary
   *                            operation
   * \param backend             The backend the Operator uses
   * \param convEventVector     Vector of dependent events
   * \return                    SNNStatus with the multiply addition event and
   * status
   */
  SNNStatus mulAdd(sycldnn::binaryop::BinaryParams params, Backend& backend,
                   std::vector<cl::sycl::event> convEventVector) {
    auto eventMulAlpha =
        sycldnn::binaryop::launch<ValueT, sycldnn::binaryop::Mul>(
            static_cast<const ValueT*>(this->devAlpha),
            static_cast<const ValueT*>(this->y), static_cast<ValueT*>(this->y),
            params, backend, convEventVector);

    auto eventMulBeta =
        sycldnn::binaryop::launch<ValueT, sycldnn::binaryop::Mul>(
            static_cast<const ValueT*>(this->devBeta),
            static_cast<const ValueT*>(this->yTmp),
            static_cast<ValueT*>(this->yTmp), params, backend, convEventVector);

    convEventVector.clear();
    convEventVector.push_back(eventMulAlpha.event);
    convEventVector.push_back(eventMulBeta.event);

    params.lhs_dims = {(int)this->ySize};
    return sycldnn::binaryop::launch<ValueT, sycldnn::binaryop::Add>(
        static_cast<const ValueT*>(this->y),
        static_cast<const ValueT*>(this->yTmp), static_cast<ValueT*>(this->y),
        params, backend, convEventVector);
  }

  /**
   * Initialize and prepare the memory needed by the scaling parameters
   * \param backend    The backend the Operator uses
   * \return           Returns the cl::sycl::event regarding mem preparation
   */
  cl::sycl::event constructMem(Backend& backend) {
    cl::sycl::event syclEvent;
    bool useAlpha, useBeta;
    useAlpha = false;
    useBeta = false;

    if ((isAlphaZero() && isBetaOne()) || (isAlphaOne() && isBetaZero()))
      return syclEvent;
    if (isAlphaZero() && isBetaZero() && !this->isBatchnormFwdTr) {
      syclEvent = q.memset(this->y, 0, this->ySize * sizeof(ValueT));
    } else {
      std::vector<cl::sycl::event> constructMemEvents;
      if (!isAlphaZero() && !isAlphaOne()) {
        this->devAlpha = backend.template allocate<ValueT>(1);
        constructMemEvents.push_back(
            q.memcpy(this->devAlpha, this->alpha, 1 * sizeof(ValueT)));
        useAlpha = true;
      }
      if (!isBetaZero()) {
        if (!isAlphaZero()) {
          this->yTmp = backend.template allocate<ValueT>(this->ySize);
          constructMemEvents.push_back(
              q.memcpy(this->yTmp, this->y, this->ySize * sizeof(ValueT)));
        }
        if (!isBetaOne()) {
          this->devBeta = backend.template allocate<ValueT>(1);
          constructMemEvents.push_back(
              q.memcpy(this->devBeta, this->beta, 1 * sizeof(ValueT)));
          useBeta = true;
        }
      }
      this->enableMulAdd = useAlpha && useBeta;
      syclEvent = sycldnn::helpers::multi_event_to_one(constructMemEvents, q);
    }
    return syclEvent;
  }

  /**
   * Applies the scaling parameters where needed depending in the alpha and beta
   * values
   * \param backend             The backend the Operator uses
   * \param convEventVector     Vector of dependent events
   * \return                    SNNStatus with the multiply addition event and
   * status
   */
  SNNStatus applyScaling(Backend& backend,
                         std::vector<cl::sycl::event> convEventVector) {
    sycldnn::binaryop::BinaryParams params{};
    params.lhs_dims = {(int)this->ySize};
    params.rhs_dims = {(int)this->ySize};
    SNNStatus scalingEvent;

    if ((isAlphaZero() && isBetaOne()) || (isAlphaOne() && isBetaZero()))
      return scalingEvent;
    if (isAlphaOne() && isBetaOne()) {
      scalingEvent = sycldnn::binaryop::launch<ValueT, sycldnn::binaryop::Add>(
          static_cast<const ValueT*>(this->y),
          static_cast<const ValueT*>(this->yTmp), static_cast<ValueT*>(this->y),
          params, backend, convEventVector);
    } else if (isAlphaZero() && !isBetaZero() && !isBetaOne()) {
      params.lhs_dims = {1};
      scalingEvent = sycldnn::binaryop::launch<ValueT, sycldnn::binaryop::Mul>(
          static_cast<const ValueT*>(this->devBeta),
          static_cast<const ValueT*>(this->y), static_cast<ValueT*>(this->y),
          params, backend, convEventVector);
    } else if (isBetaZero() && !isAlphaZero() && !isAlphaOne()) {
      params.lhs_dims = {1};
      scalingEvent = sycldnn::binaryop::launch<ValueT, sycldnn::binaryop::Mul>(
          static_cast<const ValueT*>(this->devAlpha),
          static_cast<const ValueT*>(this->y), static_cast<ValueT*>(this->y),
          params, backend, convEventVector);
    } else if (!isAlphaZero() && !isAlphaOne() && isBetaOne()) {
      params.lhs_dims = {1};
      scalingEvent = sycldnn::binaryop::launch<ValueT, sycldnn::binaryop::Mul>(
          static_cast<const ValueT*>(this->devAlpha),
          static_cast<const ValueT*>(this->y), static_cast<ValueT*>(this->y),
          params, backend, convEventVector);
      convEventVector.push_back(scalingEvent.event);

      params.lhs_dims = {(int)this->ySize};
      scalingEvent = sycldnn::binaryop::launch<ValueT, sycldnn::binaryop::Add>(
          static_cast<const ValueT*>(this->y),
          static_cast<const ValueT*>(this->yTmp), static_cast<ValueT*>(this->y),
          params, backend, convEventVector);

    } else if (!isBetaZero() && !isBetaOne() && isAlphaOne()) {
      params.lhs_dims = {1};
      scalingEvent = sycldnn::binaryop::launch<ValueT, sycldnn::binaryop::Mul>(
          static_cast<const ValueT*>(this->devBeta),
          static_cast<const ValueT*>(this->yTmp),
          static_cast<ValueT*>(this->yTmp), params, backend, convEventVector);
      convEventVector.push_back(scalingEvent.event);

      params.lhs_dims = {(int)this->ySize};
      scalingEvent = sycldnn::binaryop::launch<ValueT, sycldnn::binaryop::Add>(
          static_cast<const ValueT*>(this->y),
          static_cast<const ValueT*>(this->yTmp), static_cast<ValueT*>(this->y),
          params, backend, convEventVector);
    } else if (this->enableMulAdd) {
      params.lhs_dims = {1};
      scalingEvent = mulAdd(params, backend, convEventVector);
    }

    if (isAlphaZero() && isBetaZero() && this->isBatchnormFwdTr) {
      scalingEvent.event = q.memset(this->y, 0, this->ySize * sizeof(ValueT));
    }
    if (!isAlphaZero() && !isAlphaOne()) {
      scalingEvent.event = sycldnn::helpers::enqueue_free(
          this->q, std::vector<cl::sycl::event>{scalingEvent.event},
          this->devAlpha);
    }
    if (!isBetaZero()) {
      if (!isAlphaZero()) {
        scalingEvent.event = sycldnn::helpers::enqueue_free(
            this->q, std::vector<cl::sycl::event>{scalingEvent.event},
            this->yTmp);
      }
      if (!isBetaOne()) {
        scalingEvent.event = sycldnn::helpers::enqueue_free(
            this->q, std::vector<cl::sycl::event>{scalingEvent.event},
            this->devBeta);
      }
    }
    return scalingEvent;
  }

  /**
   * Default destructor of the class
   */
  ~ScalingParams() {}
};

}  // namespace compat

}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_COMPAT_SCALING_HPP

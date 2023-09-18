/*
 * Copyright Codeplay Software Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
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
#ifndef PORTDNN_INCLUDE_STATUS_H_
#define PORTDNN_INCLUDE_STATUS_H_

/**
 * \file
 * Contains the declarations of the \ref sycldnn::StatusCode and
 * \ref sycldnn::SNNStatus types.
 * These types are used to provide error codes and synchronization events for
 * portDNN kernel launches.
 */
#include <CL/sycl.hpp>

#include "portdnn/helpers/macros.h"
namespace sycldnn {
/** The possible errors returned by portDNN kernel launchers. */
enum class StatusCode {
  /** No error when submitting the kernel. */
  OK,
  /** An invalid algorithm was chosen for the kernel parameters. */
  InvalidAlgorithm,
  /** The tensor indices are too large for the index types. */
  IndexExceeded,
  /** The workspace buffer is too small for the chosen algorithm. */
  InsufficientWorkspace,
  /** A sufficient workspace buffer cannot be allocated on the SYCL device. */
  AllocationProblem,
  /** An invalid parameter was passed to a kernel launcher. */
  InvalidParameter,
};
/**
 * A status object containing the SYCL event corresponding to the last kernel
 * launch and a StatusCode which gives the cause of any possible error when
 * launching the kernel.
 */
struct SNNStatus {
  /**
   * \brief Construct a new SNNStatus object
   *
   * \param e SYCL event
   * \param s StatusCode
   */
  SNNStatus(const cl::sycl::event e, StatusCode s) : event(e), status(s) {}

  /**
   * \brief Construct a new SNNStatus object
   * Implicit constructor to simplify returning an error without a SYCL event.
   *
   * \param s StatusCode
   */
  SNNStatus(StatusCode s) : SNNStatus({}, s) {}

  /**
   * \brief Construct a new SNNStatus object with the OK status.
   */
  SNNStatus() : SNNStatus(StatusCode::OK) {}

  /** Default copy constructor and assignment. */
  SNN_DEFAULT_COPY(SNNStatus);

  /**
   * A SYCL event corresponding to the final SYCL kernel launch. This event can
   * be used to facilitate synchronization between the host processor and the
   * asynchronously executing OpenCL kernels that implement portDNN operators.
   *
   * This event is only valid for a successful launch, i.e. when status == OK
   */
  cl::sycl::event event;

  /**
   * A status code indicating whether the operator was launched successfully, or
   * the reason for an unsuccessful launch.
   */
  StatusCode status;
};
}  // namespace sycldnn
#endif  // PORTDNN_INCLUDE_STATUS_H_

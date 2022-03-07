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

#ifndef SYCLDNN_INCLUDE_ROI_ALIGN_LAUNCH_H_
#define SYCLDNN_INCLUDE_ROI_ALIGN_LAUNCH_H_

/**
 * \file
 * Implements the \ref sycldnn::roi_align::launch() function, which
 * asynchronously dispatches the SYCL kernels to compute a 2D pooling operation.
 */

#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include "sycldnn/helpers/macros.h"

#include "sycldnn/roi_align/params.h"

#include "sycldnn/internal/roi_align/launch_internal.h"

namespace sycldnn {
/** Namespace containing all ROI Align operations. */
namespace roi_align {
/** Namespace containing internal implementation details for pooling. */
namespace internal {

/**
 * Validate that the user provided ROI Align parameters are consistent with what
 * is expected by SYCL-DNN.
 *
 * If compiled with asserts, any invalid parameter will fail an assert.
 * Otherwise a status code \ref StatusCode::InvalidParameter will be returned.
 *
 * \param [in] params User provided parameters to validate
 * \return An SNNStatus object containing either \ref StatusCode::OK if all
 *         parameters are valid, or \ref StatusCode::InvalidParameter otherwise.
 */
SNNStatus inline validate_params(RoiAlignParams const& params) {
  SNN_VALIDATE_PARAM(params.batch > 0, "The batch size must be positive.");
  SNN_VALIDATE_PARAM(params.channels > 0,
                     "The number of channels must be positive.");
  SNN_VALIDATE_PARAM(params.in_height > 0,
                     "The number of input rows must be positive.");
  SNN_VALIDATE_PARAM(params.in_width > 0,
                     "The number of input columns must be positive.");
  SNN_VALIDATE_PARAM(params.out_height > 0,
                     "The number of output rows must be positive.");
  SNN_VALIDATE_PARAM(params.out_width > 0,
                     "The number of output columns must be positive.");
  SNN_VALIDATE_PARAM(params.num_rois > 0,
                     "The value of 'num_rois' must be positive.");
  SNN_VALIDATE_PARAM(params.sampling_ratio >= 0,
                     "The value of 'sampling_ratio' must be non-negative.");
  SNN_VALIDATE_PARAM(params.input_format == sycldnn::DataFormat::NCHW,
                     "Currently ROI Align only supports the NCHW data format.");
  // TODO(svet): This mode was added in ONNX opset 16 and is currently not
  // supported in onnxruntime. Remove this check & add tests when onnxruntime
  // adds support for this mode.
  SNN_VALIDATE_PARAM(
      params.coordinate_transformation_mode ==
          sycldnn::roi_align::CoordinateTransformationMode::OUTPUT_HALF_PIXEL,
      "Currently ROI Align only supports the 'OUTPUT_HALF_PIXEL' coordinate "
      "transformation mode.");

  return StatusCode::OK;
}

}  // namespace internal

/**
 * Launch the ROI Align operation kernel.
 *
 * \tparam T              The data type of the input and rois tensors.
 * \tparam BatchIndicesT  The type of the batch indices tensor.
 * \tparam PoolType       The type of pooling used depends on the PoolType
 *                        template parameter, which can be used to specify
 *                        either Max or Average pooling.
 * \tparam Backend        The type of the Backend.
 *
 * \param [in]  input           A pointer to the input tensor.
 * \param [in]  rois            A pointer to the ROIs tensor.
 * \param [in]  batch_indices   A pointer to the batch indices tensor.
 * \param [out] output          A pointer to the output tensor.
 * \param [in]  rap             The parameters of the ROI Align operation.
 * \param [in]  backend         The backend that provides access to the SYCL
 *                              buffers corresponding to the input and output
 *                              pointers.
 *
 * \return An \ref SNNStatus containing the SYCL event tied to the kernel
 * launches and a \ref StatusCode enum showing if the launch was OK or whether
 * it encountered some problem.
 */
template <typename T, typename BatchIndicesT,
          template <typename> class PoolType, typename Backend>
SNNStatus launch(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T const> rois,
    typename Backend::template pointer_type<BatchIndicesT const> batch_indices,
    typename Backend::template pointer_type<T> output,
    const RoiAlignParams& rap, Backend& backend) {
  auto validation_status = internal::validate_params(rap);
  if (validation_status.status != StatusCode::OK) {
    return validation_status;
  }

  auto inp_mem = backend.get_mem_object(
      input, rap.batch * rap.channels * rap.in_height * rap.in_width);
  auto rois_mem = backend.get_mem_object(rois, rap.num_rois * rap.roi_cols);
  auto batch_indices_mem = backend.get_mem_object(batch_indices, rap.num_rois);
  auto outp_mem = backend.get_mem_object(
      output, rap.num_rois * rap.channels * rap.out_height * rap.out_width);
  auto queue = backend.get_queue();

  return internal::launch_roi_align<T, BatchIndicesT, PoolType>(
      inp_mem, rois_mem, batch_indices_mem, outp_mem, rap, queue);
}

}  // namespace roi_align
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_ROI_ALIGN_LAUNCH_H_

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
#ifndef PORTDNN_INCLUDE_INTERNAL_CONV2D_BATCH_INFO_H_
#define PORTDNN_INCLUDE_INTERNAL_CONV2D_BATCH_INFO_H_

#include "portdnn/helpers/ratio.h"

namespace sycldnn {
namespace conv2d {
namespace internal {

/**
 * Struct containing info about how to split a set of images into batches.
 *
 * The total number of images will be split into `n_batches` of work, with
 * `images_per_batch` images to be computed in each batch. The total number of
 * images may not divide the number of batches, so the `last_batch_size` may be
 * different to `images_per_batch`.
 */
struct BatchInfo {
  /** Number of images per batch. */
  size_t images_per_batch;
  /** Total number of batches required. */
  size_t n_batches;
  /** Number of images in the last batch. */
  size_t last_batch_size;
};

/**
 * Get the number of batches needed to split work up into a given size of
 * minibatch.
 *
 * \param minibatch_size Size of each mini-batch.
 * \param n_images       The total number of images to process.
 * \return A BatchInfo struct containing info on how to process the images.
 */
inline BatchInfo get_batch_info(size_t minibatch_size, size_t n_images) {
  size_t const n_batches =
      helpers::round_ratio_up_above_zero(n_images, minibatch_size);
  size_t const last_batch_size = n_images - minibatch_size * (n_batches - 1);

  return BatchInfo{minibatch_size, n_batches, last_batch_size};
}

/**
 * Get the number of batches needed to spread work over a number of images
 * given a transform buffer of fixed size.
 *
 * \param buffer_size    Size of the limiting buffer.
 * \param n_images       The total number of images to process.
 * \param size_per_image The size in the limiting buffer required by a image.
 * \return A BatchInfo struct containing info on how to process the images.
 */
inline BatchInfo get_batch_info(size_t buffer_size, size_t n_images,
                                size_t size_per_image) {
  // The number of images per batch is bounded by the total number of images
  size_t const images_per_buffer =
      std::min(n_images, buffer_size / size_per_image);
  size_t const n_batches =
      helpers::round_ratio_up_above_zero(n_images, images_per_buffer);
  size_t const minibatch_size =
      helpers::round_ratio_up_above_zero(n_images, n_batches);
  size_t const last_batch_size = n_images - minibatch_size * (n_batches - 1);

  return BatchInfo{minibatch_size, n_batches, last_batch_size};
}

}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_INTERNAL_CONV2D_BATCH_INFO_H_

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

#ifndef PORTDNN_TEST_HELPERS_TRANSPOSE_H_
#define PORTDNN_TEST_HELPERS_TRANSPOSE_H_

#include <vector>

/**
 * \brief Transposes NXY to NYX.
 *
 * \param output
 * \param input
 * \param N Batch size
 * \param X
 * \param Y
 * \param offset Optional input's offset will be untouched.
 */
template <typename T>
void transpose(std::vector<T>& output, const std::vector<T>& input, size_t N,
               size_t X, size_t Y, size_t offset = 0) {
  output.resize(input.size(), T(0));
  assert(N * X * Y + offset <= input.size());
  assert(N * X * Y + offset <= output.size());
  for (size_t n = 0; n < N; ++n) {
    for (size_t x = 0; x < X; ++x) {
      for (size_t y = 0; y < Y; ++y) {
        size_t out_idx = ((n * Y) + y) * X + x + offset;
        size_t in_idx = ((n * X) + x) * Y + y + offset;
        output[out_idx] = input[in_idx];
      }
    }
  }
}

#endif  // PORTDNN_TEST_HELPERS_TRANSPOSE_H_

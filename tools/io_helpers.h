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

#include <string>
#include <vector>

#include <fstream>
#include <iostream>

namespace sycldnn {
namespace helpers {
// Helper function that reads binary data produced by h5tobin.py into a vector
std::vector<char> read_binary_data(std::string const& name) {
  std::ifstream file(name, std::ios_base::binary | std::ios_base::in);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file " + name);
  }
  std::vector<char> output{std::istreambuf_iterator<char>{file}, {}};
  return output;
}

// read image data from disk
template <typename DType, typename DeviceMem, typename Backend>
DeviceMem read_image_data(std::string const& name, Backend& backend,
                          size_t size) {
  cl::sycl::range<1> r{size};  // resnet input size
  cl::sycl::buffer<DType> b{r};
  auto data = read_binary_data(name);
  assert(data.size() == size * sizeof(DType));
  {
    auto char_data = b.template reinterpret<char>(r * sizeof(DType));
    auto event = backend.get_queue().submit([&](cl::sycl::handler& cgh) {
      auto acc =
          char_data.template get_access<cl::sycl::access::mode::discard_write>(
              cgh);
      cgh.copy(data.data(), acc);
    });
    event.wait_and_throw();
  }
  return DeviceMem{b, 0};
}

}  // namespace helpers
}  // namespace sycldnn
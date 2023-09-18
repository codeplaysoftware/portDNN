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
#ifndef PORTDNN_BENCH_FIXTURE_ADD_SYCL_DEVICE_INFO_H_
#define PORTDNN_BENCH_FIXTURE_ADD_SYCL_DEVICE_INFO_H_

#include "string_reporter.h"

#include <string>

#include <CL/sycl.hpp>

namespace sycldnn {
namespace bench {
namespace device_info {

/**
 * Add device info from the provided SYCL device to the benchmark label.
 *
 * \param [in] device    SYCL device to query for info to add to the label.
 * \param [out] reporter The benchmark string reporter to add the info to.
 */
inline void add_opencl_device_info(cl::sycl::device const& device,
                                   StringReporter& reporter) {
  // OpenCL is unclear whether strings returned from clGet*Info() should be
  // null terminated, and ComputeCpp currently copies embedded nulls.
  // On some OpenCL implementations this results in strings that behave
  // unexpectedly when appended to. This lambda trims those strings.
  auto trim = [](std::string s) -> std::string {
    s.resize(strlen(s.c_str()));
    return s;
  };
  auto device_name = device.get_info<cl::sycl::info::device::name>();
  auto device_version = device.get_info<cl::sycl::info::device::version>();
  auto vendor_name = device.get_info<cl::sycl::info::device::vendor>();
  auto driver_version =
      device.get_info<cl::sycl::info::device::driver_version>();

  reporter.add_to_label("device_name", trim(device_name));
  reporter.add_to_label("device_version", trim(device_version));
  reporter.add_to_label("vendor_name", trim(vendor_name));
  reporter.add_to_label("driver_version", trim(driver_version));
}

}  // namespace device_info
}  // namespace bench
}  // namespace sycldnn

#endif  // PORTDNN_BENCH_FIXTURE_ADD_SYCL_DEVICE_INFO_H_

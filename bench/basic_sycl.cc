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
#include <benchmark/benchmark.h>

#include <stddef.h>
#include <utility>
#include <vector>

#include <CL/sycl.hpp>

template <typename T>
struct SetBuffer {
  static constexpr auto write_mode = cl::sycl::access::mode::discard_write;
  static constexpr auto global_access = cl::sycl::access::target::global_buffer;
  using write_accessor = cl::sycl::accessor<T, 1, write_mode, global_access>;

  explicit SetBuffer(write_accessor out) : output(std::move(out)) {}

  void operator()(cl::sycl::item<1> item) const {
    const auto id = item.get_id(0);
    output[id] = id * static_cast<T>(0.1);
  }

 private:
  write_accessor output;
};
static void BM_SetHostBufferDestruct(benchmark::State& state) {
  static constexpr auto write_mode = SetBuffer<float>::write_mode;
  const size_t num_elems = state.range(0);
  cl::sycl::default_selector selector;
  cl::sycl::queue queue{selector};
  std::vector<float> base_mem;
  base_mem.reserve(num_elems);
  for (auto _ : state) {
    cl::sycl::buffer<float, 1> bufA{base_mem.data(),
                                    cl::sycl::range<1>{num_elems}};
    auto ev = queue.submit([&](cl::sycl::handler& cgh) {
      auto accessor = bufA.template get_access<write_mode>(cgh);
      SetBuffer<float> functor(accessor);
      cgh.parallel_for(cl::sycl::range<1>{num_elems}, functor);
    });
    ev.wait_and_throw();
  }
}
static void BM_SetDeviceBufferDestruct(benchmark::State& state) {
  static constexpr auto write_mode = SetBuffer<float>::write_mode;
  const size_t num_elems = state.range(0);
  cl::sycl::default_selector selector;
  cl::sycl::queue queue{selector};
  for (auto _ : state) {
    cl::sycl::buffer<float, 1> bufA{cl::sycl::range<1>{num_elems}};
    auto ev = queue.submit([&](cl::sycl::handler& cgh) {
      auto accessor = bufA.template get_access<write_mode>(cgh);
      SetBuffer<float> functor(accessor);
      cgh.parallel_for(cl::sycl::range<1>{num_elems}, functor);
    });
    ev.wait_and_throw();
  }
}
static void BM_SetDeviceBufferNoDestruct(benchmark::State& state) {
  static constexpr auto write_mode = SetBuffer<float>::write_mode;
  const size_t num_elems = state.range(0);
  cl::sycl::default_selector selector;
  cl::sycl::queue queue{selector};
  cl::sycl::buffer<float, 1> bufA{cl::sycl::range<1>{num_elems}};
  for (auto _ : state) {
    auto ev = queue.submit([&](cl::sycl::handler& cgh) {
      auto accessor = bufA.template get_access<write_mode>(cgh);
      SetBuffer<float> functor(accessor);
      cgh.parallel_for(cl::sycl::range<1>{num_elems}, functor);
    });
    ev.wait_and_throw();
  }
}
BENCHMARK(BM_SetHostBufferDestruct)->Range(8 << 4, 8 << 10);
BENCHMARK(BM_SetDeviceBufferDestruct)->Range(8 << 4, 8 << 10);
BENCHMARK(BM_SetDeviceBufferNoDestruct)->Range(8 << 4, 8 << 10);

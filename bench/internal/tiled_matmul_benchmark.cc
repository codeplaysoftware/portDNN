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

#include "portdnn/helpers/handle_exception.h"
#include "portdnn/helpers/scope_exit.h"

#include "portdnn/backend/snn_backend.h"

#include "portdnn/matmul/params.h"
#include "src/backend/backend_provider.h"
#include "src/backend/snn_backend_provider.h"
#include "src/matmul/queue_kernel.h"

#include "bench/fixture/add_computecpp_info.h"
#include "bench/fixture/add_datatype_info.h"
#include "bench/fixture/add_sycl_device_info.h"
#include "bench/fixture/base_executor.h"
#include "bench/fixture/statistic.h"
#include "bench/fixture/string_reporter.h"
#include "bench/fixture/typenames.h"

#include "bench/matmul/benchmark_params.h"

#include <iostream>
#include <vector>

#ifndef CSV_IO_NO_THREAD
#define CSV_IO_NO_THREAD 1
#endif
#include "csv.h"

namespace sycldnn {
namespace bench {

/** Helper function that checks if portDNN can wait on events directly, or
 * has to wait on the queue. This is because Eigen cannot return us the events
 * corresponding to the kernel launch directly. */
/* TODO: SD-404 Remove queue::wait_and_throw workaround when Eigen removed */
inline void wait_for_event(cl::sycl::event& ev, cl::sycl::queue q) {
  if (ev.is_host()) {
    q.wait_and_throw();
  } else {
    ev.wait_and_throw();
  }
}

/** Executor to perform a matrix multiply benchmark using portDNN.  */
template <typename Benchmark, typename DataType, int RowTile, int AccTile,
          int ColTile>
struct SNNMatmulExecutor : public BaseExecutor {
 private:
  using State = ::benchmark::State;

  /** Get a reference to the underlying benchmark fixture. */
  Benchmark& underlying_benchmark() { return static_cast<Benchmark&>(*this); }

 public:
  /** Execute a conv2d benchmark with the given parameters and selector. */
  void execute(State& state, int m, int k, int n, int batch, int workgroup_rows,
               int workgroup_cols, int workgroup_batch) {
    auto& benchmark = underlying_benchmark();
    auto& backend = benchmark.get_backend();
    auto& queue = backend.get_queue();

    using Backend = typename std::remove_reference<decltype(backend)>::type;
    using Pointer = typename Backend::template internal_pointer_type<DataType>;
    using ConstPointer =
        typename Backend::template internal_pointer_type<const DataType>;

    auto lhs_size = batch * m * k;
    auto rhs_size = batch * k * n;
    auto out_size = batch * m * n;

    std::vector<float> lhs_vec(lhs_size);
    std::vector<float> rhs_vec(rhs_size);
    std::vector<float> out_vec(out_size);

    Pointer lhs_gpu;
    Pointer rhs_gpu;
    Pointer out_gpu;
    SNN_ON_SCOPE_EXIT {
      benchmark.deallocate_ptr(out_gpu);
      benchmark.deallocate_ptr(rhs_gpu);
      benchmark.deallocate_ptr(lhs_gpu);
    };

    try {
      lhs_gpu =
          benchmark.get_initialised_device_memory(lhs_vec.size(), lhs_vec);
      rhs_gpu =
          benchmark.get_initialised_device_memory(rhs_vec.size(), rhs_vec);
      out_gpu =
          benchmark.get_initialised_device_memory(out_vec.size(), out_vec);
    } catch (cl::sycl::exception const& e) {
      helpers::handle_exception(e, [&](std::string& msg) {
        state.SkipWithError((msg + UnexpectedFailure).c_str());
      });
      return;
    } catch (std::exception const& e) {
      helpers::handle_exception(e, [&](std::string& msg) {
        state.SkipWithError((msg + UnexpectedFailure).c_str());
      });
      return;
    }

    ConstPointer const_lhs_gpu = lhs_gpu;
    ConstPointer const_rhs_gpu = rhs_gpu;

    auto lhs_mem = backend.get_mem_object(const_lhs_gpu, lhs_vec.size());
    auto rhs_mem = backend.get_mem_object(const_rhs_gpu, rhs_vec.size());
    auto out_mem = backend.get_mem_object(out_gpu, out_vec.size());

    {  // Ensure the kernel is built before benchmarking
      SNNStatus status;
      try {
        status =
            ((m % RowTile == 0) && (k % AccTile == 0) && (n % ColTile == 0))
                ? matmul::internal::queue_kernel<DataType, int32_t, false,
                                                 false, RowTile, AccTile,
                                                 ColTile, false>(
                      lhs_mem, rhs_mem, out_mem,
                      sycldnn::matmul::MatmulParams{batch, m, k, n, 0.f}, queue,
                      workgroup_rows, workgroup_cols, workgroup_batch, {})
                : matmul::internal::queue_kernel<DataType, int32_t, false,
                                                 false, RowTile, AccTile,
                                                 ColTile, true>(
                      lhs_mem, rhs_mem, out_mem,
                      sycldnn::matmul::MatmulParams{batch, m, k, n, 0.f}, queue,
                      workgroup_rows, workgroup_cols, workgroup_batch, {});

      } catch (cl::sycl::exception const& e) {
        helpers::handle_exception(e, [&](std::string& msg) {
          state.SkipWithError((msg + UnexpectedFailure).c_str());
        });
        return;
      }

      if (sycldnn::StatusCode::OK != status.status) {
        state.SkipWithError(UnsupportedFailure);
        return;
      }

      try {
        wait_for_event(status.event, queue);
      } catch (cl::sycl::exception const& e) {
        helpers::handle_exception(e, [&](std::string& msg) {
          state.SkipWithError((msg + UnexpectedFailure).c_str());
        });
        return;
      } catch (std::exception const& e) {
        helpers::handle_exception(e, [&](std::string& msg) {
          state.SkipWithError((msg + UnexpectedFailure).c_str());
        });
        return;
      }
    }

    for (auto _ : state) {
      this->start_timing();
      try {
        auto status =
            ((m % RowTile == 0) && (k % AccTile == 0) && (n % ColTile == 0))
                ? matmul::internal::queue_kernel<DataType, int32_t, false,
                                                 false, RowTile, AccTile,
                                                 ColTile, false>(
                      lhs_mem, rhs_mem, out_mem,
                      sycldnn::matmul::MatmulParams{batch, m, k, n, 0.f}, queue,
                      workgroup_rows, workgroup_cols, workgroup_batch, {})
                : matmul::internal::queue_kernel<DataType, int32_t, false,
                                                 false, RowTile, AccTile,
                                                 ColTile, true>(
                      lhs_mem, rhs_mem, out_mem,
                      sycldnn::matmul::MatmulParams{batch, m, k, n, 0.f}, queue,
                      workgroup_rows, workgroup_cols, workgroup_batch, {});
        wait_for_event(status.event, backend.get_queue());
      } catch (cl::sycl::exception const& e) {
        helpers::handle_exception(e, [&](std::string& msg) {
          state.SkipWithError((msg + UnexpectedFailure).c_str());
        });
        return;
      } catch (std::exception const& e) {
        helpers::handle_exception(e, [&](std::string& msg) {
          state.SkipWithError((msg + UnexpectedFailure).c_str());
        });
        return;
      }

      this->end_timing();
      this->set_iteration_time(state);
    }

    state.SetItemsProcessed(state.iterations() * 2 * batch * m * k * n);
    state.counters["workgroup_rows"] = workgroup_rows;
    state.counters["workgroup_cols"] = workgroup_cols;
    state.counters["workgroup_batch"] = workgroup_batch;
    state.counters["m"] = m;
    state.counters["k"] = k;
    state.counters["n"] = n;
    state.counters["batch"] = batch;
    state.counters["row_tile"] = RowTile;
    state.counters["acc_tile"] = AccTile;
    state.counters["col_tile"] = ColTile;

    this->finish_benchmark(state);
  }
};
}  // namespace bench
}  // namespace sycldnn

extern const char* commit_date;
extern const char* commit_hash;

template <typename Backend, typename DataType, int RowTile, int AccTile,
          int ColTile>
class SNNMatmulBenchmark
    : public sycldnn::bench::SNNMatmulExecutor<
          SNNMatmulBenchmark<Backend, DataType, RowTile, AccTile, ColTile>,
          DataType, RowTile, AccTile, ColTile>,
      public sycldnn::backend::BackendProvider<Backend>,
      public sycldnn::bench::StringReporter,
      public benchmark::Fixture {
 private:
  using State = benchmark::State;

 protected:
  void BenchmarkCase(State& state) override {
    auto params = matmul_benchmark_params::deserialize(state);
    this->add_statistic(std::unique_ptr<sycldnn::bench::Statistic>{
        new sycldnn::bench::MaxStatistic{}});
    this->add_statistic(std::unique_ptr<sycldnn::bench::Statistic>{
        new sycldnn::bench::MinStatistic{}});
    this->add_statistic(std::unique_ptr<sycldnn::bench::Statistic>{
        new sycldnn::bench::StdDevStatistic{}});
    this->execute(state, params.m, params.k, params.n, params.batch,
                  state.range(4), state.range(5), state.range(6));

    // Get the SYCL device, and add device and driver info to the benchmark.
    auto& backend = this->get_backend();
    auto dev = backend.get_queue().get_device();
    sycldnn::bench::device_info::add_opencl_device_info(dev, *this);
    sycldnn::bench::computecpp_info::add_computecpp_version(*this);
    sycldnn::bench::datatype_info::add_datatype_info<DataType>(*this);

    this->add_to_label("@library", "portDNN");
    this->add_to_label("@backend", backend.name());
    this->add_to_label("short_name", "Matmul");
    this->add_to_label("git_hash", commit_hash);
    this->set_label(state);
  }

  void set_model(const char* model_name) {
    this->add_to_label("@model_name", model_name);
  }
};

#define MATMUL_BENCHMARK(name, ...)              \
  class SNNMatmulBenchmark##_##name##_Benchmark  \
      : public SNNMatmulBenchmark<__VA_ARGS__> { \
   public:                                       \
    SNNMatmulBenchmark##_##name##_Benchmark()    \
        : SNNMatmulBenchmark<__VA_ARGS__>() {    \
      this->SetName(#name);                      \
      this->set_model("TiledMatmul");            \
    }                                            \
  };

#define CALL_WITH_PARAMS_IMPL(MACRO, ...) MACRO(__VA_ARGS__)

#define CALL_WITH_ROW_ACC(MACRO, ROW, ACC)   \
  CALL_WITH_PARAMS_IMPL(MACRO, ROW, ACC, 1); \
  CALL_WITH_PARAMS_IMPL(MACRO, ROW, ACC, 2); \
  CALL_WITH_PARAMS_IMPL(MACRO, ROW, ACC, 4); \
  CALL_WITH_PARAMS_IMPL(MACRO, ROW, ACC, 8)

#define CALL_WITH_ROW(MACRO, ROW)   \
  CALL_WITH_ROW_ACC(MACRO, ROW, 1); \
  CALL_WITH_ROW_ACC(MACRO, ROW, 2); \
  CALL_WITH_ROW_ACC(MACRO, ROW, 4); \
  CALL_WITH_ROW_ACC(MACRO, ROW, 8)

#define CALL_WITH_PARAMS(MACRO) \
  CALL_WITH_ROW(MACRO, 1);      \
  CALL_WITH_ROW(MACRO, 2);      \
  CALL_WITH_ROW(MACRO, 4);      \
  CALL_WITH_ROW(MACRO, 8)

#define GENERATE_BENCH(ROW, ACC, COL)                 \
  MATMUL_BENCHMARK(TiledMatmul_##ROW##_##ACC##_##COL, \
                   sycldnn::backend::SNNBackend, float, ROW, ACC, COL)

CALL_WITH_PARAMS(GENERATE_BENCH);

void register_benchmark(
    std::vector<::benchmark::internal::Benchmark*> registered_benchmarks, int m,
    int k, int n, int batch) {
  for (auto bench : registered_benchmarks) {
    bench->Args({m, k, n, batch, 1, 64, 1})
        ->Args({m, k, n, batch, 1, 128, 1})
        ->Args({m, k, n, batch, 8, 8, 1})
        ->Args({m, k, n, batch, 8, 16, 1})
        ->Args({m, k, n, batch, 8, 32, 1})
        ->Args({m, k, n, batch, 16, 8, 1})
        ->Args({m, k, n, batch, 16, 16, 1})
        ->Args({m, k, n, batch, 32, 8, 1})
        ->Args({m, k, n, batch, 64, 1, 1})
        ->Args({m, k, n, batch, 128, 1, 1});
  }
}

int main(int argc, char** argv) {
  ::benchmark::Initialize(&argc, argv);
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <file> [gtest-options]\n";
    std::cerr << "File should be a CSV of matmul sizes. Options are standard "
                 "Google Test options\n";
    return 1;
  }

  char const* const csv_file = argv[1];
  std::vector<::benchmark::internal::Benchmark*> benchmarks;
#define REGISTER_BENCHMARK(ROW, ACC, COL)                                               \
  {                                                                                     \
    auto bench =                                                                        \
        ::benchmark::internal::RegisterBenchmarkInternal(                               \
            new SNNMatmulBenchmark##_##TiledMatmul_##ROW##_##ACC##_##COL##_Benchmark()) \
            ->UseManualTime()                                                           \
            ->Unit(benchmark::kNanosecond);                                             \
    benchmarks.push_back(bench);                                                        \
  }

  CALL_WITH_PARAMS(REGISTER_BENCHMARK);

  io::CSVReader<4> reader(csv_file);
  reader.read_header(io::ignore_extra_column, "M", "N", "K", "batch");
  int m;
  int n;
  int k;
  int batch;
  while (reader.read_row(m, n, k, batch)) {
    register_benchmark(benchmarks, m, k, n, batch);
  }
  ::benchmark::RunSpecifiedBenchmarks();
}

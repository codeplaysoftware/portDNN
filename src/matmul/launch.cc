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
#include "sycldnn/internal/matmul/launch.h"

#include "sycldnn/mem_object.h"

#include "src/matmul/queue_kernel.h"

namespace sycldnn {
namespace matmul {
namespace internal {
namespace {

// Launch the kernel specified by the template parameters.
template <typename T, bool TransposeLHS, bool TransposeRHS, int RowTile,
          int AccTile, int ColTile>
SNNStatus launch_with_tiles(BaseMemObject<T const>& lhs,
                            BaseMemObject<T const>& rhs,
                            BaseMemObject<T>& output, int batches, int m, int k,
                            int n, T beta, cl::sycl::queue& queue,
                            size_t wg_rows, size_t wg_cols, size_t wg_batch) {
  auto kernel = ((m % RowTile == 0) && (k % AccTile == 0) && (n % ColTile == 0))
                    ? queue_kernel<T, int, TransposeLHS, TransposeRHS, RowTile,
                                   AccTile, ColTile, false>
                    : queue_kernel<T, int, TransposeLHS, TransposeRHS, RowTile,
                                   AccTile, ColTile, true>;
  return kernel(lhs, rhs, output, batches, m, k, n, beta, queue, wg_rows,
                wg_cols, wg_batch);
}
template <typename T, bool TransposeLHS, bool TransposeRHS>
SNNStatus launch_for_intelgpu(BaseMemObject<T const>& lhs,
                              BaseMemObject<T const>& rhs,
                              BaseMemObject<T>& output, int batches, int m,
                              int k, int n, T beta, cl::sycl::queue& queue) {
#define LAUNCH(RT, AT, CT, WR, WC, WB)                          \
  launch_with_tiles<T, TransposeLHS, TransposeRHS, RT, AT, CT>( \
      lhs, rhs, output, batches, m, k, n, beta, queue, WR, WC, WB);
  if (n % 8 <= 3.5) {
    if (k * n <= 6322176) {
      if (k * n <= 107648) {
        if (n <= 56.5) {
          if ((float)n / k <= 0.05517578125) {
            if (k * n <= 12533.5) {
              return LAUNCH(4, 2, 8, 8, 8, 1);
            } else {
              return LAUNCH(4, 4, 8, 1, 64, 1);
            }
          } else {
            return LAUNCH(4, 1, 4, 1, 64, 1);
          }
        } else {
          if (k <= 114) {
            if (batches * m * n <= 6889472) {
              return LAUNCH(4, 4, 8, 1, 64, 1);
            } else {
              return LAUNCH(4, 4, 8, 16, 8, 1);
            }
          } else {
            if ((float)m / k <= 0.5765306055545807) {
              return LAUNCH(4, 4, 8, 1, 64, 1);
            } else {
              return LAUNCH(8, 4, 4, 8, 16, 1);
            }
          }
        }
      } else {
        if (batches * m * n <= 466944) {
          if (batches * m * n <= 43520) {
            if ((float)m / k <= 0.09637188166379929) {
              return LAUNCH(4, 4, 8, 1, 64, 1);
            } else {
              return LAUNCH(4, 1, 4, 1, 64, 1);
            }
          } else {
            if (k <= 2720) {
              return LAUNCH(4, 4, 8, 1, 64, 1);
            } else {
              return LAUNCH(4, 2, 4, 8, 16, 1);
            }
          }
        } else {
          if (k * m <= 3072) {
            if (k * m <= 1888) {
              return LAUNCH(4, 4, 8, 16, 8, 1);
            } else {
              return LAUNCH(4, 2, 8, 8, 8, 1);
            }
          } else {
            return LAUNCH(4, 4, 8, 1, 64, 1);
          }
        }
      }
    } else {
      if (m <= 33.5) {
        return LAUNCH(8, 4, 4, 8, 16, 1);
      } else {
        if (batches * m * n <= 55296) {
          if (n % 8 <= 1.5) {
            return LAUNCH(4, 2, 8, 8, 8, 1);
          } else {
            return LAUNCH(4, 4, 8, 1, 64, 1);
          }
        } else {
          if (k % 2 <= 0.5) {
            if ((float)m * n / k <= 2.204081654548645) {
              return LAUNCH(4, 4, 8, 1, 64, 1);
            } else {
              return LAUNCH(4, 2, 4, 8, 16, 1);
            }
          } else {
            return LAUNCH(4, 4, 8, 1, 64, 1);
          }
        }
      }
    }
  } else {
    if ((float)m / k <= 0.05603387230075896) {
      return LAUNCH(8, 4, 4, 8, 16, 1);
    } else {
      if ((float)m * n / k <= 12.88888931274414) {
        return LAUNCH(4, 1, 4, 1, 64, 1);
      } else {
        return LAUNCH(4, 2, 8, 8, 8, 1);
      }
    }
  }
#undef LAUNCH
}
template <typename T, bool TransposeLHS, bool TransposeRHS>
SNNStatus launch_for_intelcpu(BaseMemObject<T const>& lhs,
                              BaseMemObject<T const>& rhs,
                              BaseMemObject<T>& output, int batch, int m, int k,
                              int n, T beta, cl::sycl::queue& queue) {
#define LAUNCH(RT, AT, CT, WR, WC, WB)                          \
  launch_with_tiles<T, TransposeLHS, TransposeRHS, RT, AT, CT>( \
      lhs, rhs, output, batch, m, k, n, beta, queue, WR, WC, WB);
  if (k <= 82) {
    if (m % 8 <= 1.5) {
      if (k * n <= 8800) {
        if (batch * m * n <= 2916352) {
          return LAUNCH(8, 4, 8, 1, 128, 1);
        } else {
          return LAUNCH(4, 1, 1, 1, 64, 1);
        }
      } else {
        if ((float)m * n / k <= 773082.09375) {
          if (n % 8 <= 0.5) {
            return LAUNCH(4, 1, 8, 8, 16, 1);
          } else {
            if (m * n <= 985680) {
              return LAUNCH(8, 4, 8, 1, 128, 1);
            } else {
              return LAUNCH(4, 1, 8, 8, 16, 1);
            }
          }
        } else {
          return LAUNCH(4, 1, 1, 1, 64, 1);
        }
      }
    } else {
      return LAUNCH(4, 1, 1, 1, 64, 1);
    }
  } else {
    if (batch * m * n <= 861184) {
      if ((float)m / k <= 0.03148849494755268) {
        if (batch * m * n <= 139264) {
          if (m * n <= 1888) {
            if ((float)m * n / k <= 0.006179932039231062) {
              return LAUNCH(8, 4, 8, 1, 128, 1);
            } else {
              return LAUNCH(4, 1, 1, 1, 64, 1);
            }
          } else {
            if (k * n <= 1522816) {
              return LAUNCH(8, 4, 8, 128, 1, 1);
            } else {
              return LAUNCH(4, 1, 1, 1, 64, 1);
            }
          }
        } else {
          return LAUNCH(4, 1, 1, 1, 64, 1);
        }
      } else {
        if (n <= 544) {
          if (k * n <= 39488) {
            if (n % 8 <= 0.5) {
              return LAUNCH(8, 4, 8, 64, 1, 1);
            } else {
              return LAUNCH(8, 4, 8, 1, 128, 1);
            }
          } else {
            return LAUNCH(8, 4, 8, 128, 1, 1);
          }
        } else {
          if ((float)n / k <= 17.01388931274414) {
            if (k * m <= 231424) {
              return LAUNCH(8, 1, 1, 1, 64, 1);
            } else {
              return LAUNCH(8, 4, 8, 16, 8, 1);
            }
          } else {
            if (k % 4 <= 1.5) {
              return LAUNCH(8, 4, 8, 64, 1, 1);
            } else {
              return LAUNCH(8, 4, 8, 8, 8, 1);
            }
          }
        }
      }
    } else {
      if (batch * m * n <= 2785280) {
        if (k <= 1874) {
          if ((float)m / k <= 0.8545706272125244) {
            if ((float)n / k <= 0.35467155277729034) {
              return LAUNCH(8, 4, 8, 64, 1, 1);
            } else {
              return LAUNCH(8, 4, 8, 8, 8, 1);
            }
          } else {
            if (k * n <= 18816) {
              return LAUNCH(8, 4, 8, 1, 128, 1);
            } else {
              return LAUNCH(8, 4, 8, 64, 1, 1);
            }
          }
        } else {
          if ((float)m / k <= 0.13718820735812187) {
            return LAUNCH(8, 4, 8, 16, 8, 1);
          } else {
            if (k * m <= 1392640) {
              return LAUNCH(8, 4, 8, 16, 8, 1);
            } else {
              return LAUNCH(8, 1, 1, 1, 64, 1);
            }
          }
        }
      } else {
        if ((float)n / k <= 18.375) {
          if (batch * m * n <= 3244032) {
            if ((float)m * n / k <= 1088.888916015625) {
              return LAUNCH(8, 1, 1, 1, 64, 1);
            } else {
              return LAUNCH(8, 4, 8, 8, 8, 1);
            }
          } else {
            return LAUNCH(8, 4, 8, 8, 8, 1);
          }
        } else {
          if ((float)n / k <= 92.55555725097656) {
            return LAUNCH(8, 4, 8, 64, 1, 1);
          } else {
            if (m * n <= 8028160) {
              return LAUNCH(8, 4, 8, 8, 8, 1);
            } else {
              return LAUNCH(4, 1, 1, 1, 64, 1);
            }
          }
        }
      }
    }
  }

#undef LAUNCH
}
template <typename T, bool TransposeLHS, bool TransposeRHS>
SNNStatus launch_for_amd(BaseMemObject<T const>& lhs,
                         BaseMemObject<T const>& rhs, BaseMemObject<T>& output,
                         int batch, int m, int k, int n, T beta,
                         cl::sycl::queue& queue) {
#define LAUNCH(RT, AT, CT, WR, WC, WB)                          \
  launch_with_tiles<T, TransposeLHS, TransposeRHS, RT, AT, CT>( \
      lhs, rhs, output, batch, m, k, n, beta, queue, WR, WC, WB);

  if (batch * m * n <= 243968) {
    if (batch * m * n <= 34816) {
      if (n % 4 <= 0.5) {
        if (k * n <= 75264) {
          return LAUNCH(1, 8, 4, 8, 8, 1);
        } else {
          return LAUNCH(4, 8, 4, 8, 32, 1);
        }
      } else {
        if ((float)m / k <= 0.0004840877518290654) {
          return LAUNCH(4, 8, 4, 8, 32, 1);
        } else {
          return LAUNCH(1, 8, 4, 8, 8, 1);
        }
      }
    } else {
      if (batch * m * n <= 69632) {
        if (k * n <= 18816) {
          return LAUNCH(1, 8, 4, 8, 8, 1);
        } else {
          if (m <= 768) {
            return LAUNCH(4, 8, 4, 8, 32, 1);
          } else {
            if (k * m <= 1572864) {
              return LAUNCH(4, 8, 4, 8, 32, 1);
            } else {
              return LAUNCH(1, 8, 4, 8, 8, 1);
            }
          }
        }
      } else {
        if (n <= 32.5) {
          return LAUNCH(1, 8, 4, 8, 8, 1);
        } else {
          if (k * n <= 71296) {
            if (m * n <= 165888) {
              return LAUNCH(4, 8, 4, 8, 32, 1);
            } else {
              return LAUNCH(1, 8, 2, 16, 16, 1);
            }
          } else {
            if (batch * m * n <= 139264) {
              return LAUNCH(1, 8, 2, 16, 16, 1);
            } else {
              return LAUNCH(4, 8, 4, 8, 32, 1);
            }
          }
        }
      }
    }
  } else {
    if ((float)m / k <= 4.559999942779541) {
      if (k * m <= 6144) {
        if (m <= 17.5) {
          return LAUNCH(4, 8, 4, 8, 32, 1);
        } else {
          if (batch * m * n <= 1591360) {
            if ((float)n / k <= 7.65625) {
              return LAUNCH(4, 8, 4, 8, 32, 1);
            } else {
              return LAUNCH(1, 8, 1, 16, 16, 1);
            }
          } else {
            if (batch * m * n <= 10035200) {
              return LAUNCH(2, 8, 4, 16, 16, 1);
            } else {
              return LAUNCH(4, 8, 4, 8, 32, 1);
            }
          }
        }
      } else {
        if (n <= 96) {
          if (n <= 56.5) {
            return LAUNCH(4, 8, 4, 8, 32, 1);
          } else {
            return LAUNCH(1, 8, 2, 16, 16, 1);
          }
        } else {
          if (batch * m * n <= 278528) {
            if ((float)m / k <= 0.48979590833187103) {
              return LAUNCH(2, 8, 4, 16, 16, 1);
            } else {
              return LAUNCH(2, 4, 4, 8, 32, 1);
            }
          } else {
            return LAUNCH(2, 8, 4, 16, 16, 1);
          }
        }
      }
    } else {
      if (batch * m * n <= 1867776) {
        if (k * m <= 37632) {
          return LAUNCH(1, 8, 1, 16, 16, 1);
        } else {
          if ((float)m * n / k <= 4012.4080810546875) {
            return LAUNCH(1, 8, 1, 16, 16, 1);
          } else {
            if (k * n <= 37632) {
              return LAUNCH(1, 8, 1, 16, 16, 1);
            } else {
              return LAUNCH(4, 8, 4, 8, 32, 1);
            }
          }
        }
      } else {
        if (k * n <= 213248) {
          return LAUNCH(4, 8, 4, 8, 32, 1);
        } else {
          return LAUNCH(1, 8, 1, 16, 16, 1);
        }
      }
    }
  }
#undef LAUNCH
}
template <typename T, bool TransposeLHS, bool TransposeRHS>
SNNStatus launch_for_arm(BaseMemObject<T const>& lhs,
                         BaseMemObject<T const>& rhs, BaseMemObject<T>& output,
                         int batch, int m, int k, int n, T beta,
                         cl::sycl::queue& queue) {
#define LAUNCH(RT, AT, CT, WR, WC, WB)                          \
  launch_with_tiles<T, TransposeLHS, TransposeRHS, RT, AT, CT>( \
      lhs, rhs, output, batch, m, k, n, beta, queue, WR, WC, WB);
  if ((float)m / k <= 0.27437641471624374) {
    if (m * n <= 12896) {
      if (batch * m * n <= 83968) {
        return LAUNCH(2, 4, 2, 8, 32, 1);
      } else {
        if ((float)m / k <= 0.06775882840156555) {
          return LAUNCH(4, 2, 4, 8, 16, 1);
        } else {
          return LAUNCH(2, 4, 2, 8, 32, 1);
        }
      }
    } else {
      if ((float)n / k <= 4.083333492279053) {
        if (k * m <= 4816896) {
          return LAUNCH(4, 2, 2, 1, 64, 1);
        } else {
          if (m * n <= 110592) {
            return LAUNCH(2, 4, 2, 8, 32, 1);
          } else {
            return LAUNCH(4, 2, 2, 1, 64, 1);
          }
        }
      } else {
        if (k * m <= 19296) {
          if (k * n <= 72253440) {
            if (m * n <= 94080) {
              return LAUNCH(4, 2, 4, 8, 16, 1);
            } else {
              return LAUNCH(4, 2, 2, 1, 64, 1);
            }
          } else {
            return LAUNCH(4, 2, 4, 8, 16, 1);
          }
        } else {
          return LAUNCH(2, 4, 2, 8, 32, 1);
        }
      }
    }
  } else {
    if (k * m <= 960) {
      return LAUNCH(4, 2, 4, 8, 16, 1);
    } else {
      if ((float)n / k <= 0.1103515625) {
        if ((float)n / k <= 0.0634765625) {
          return LAUNCH(4, 2, 4, 1, 128, 1);
        } else {
          return LAUNCH(4, 2, 4, 8, 16, 1);
        }
      } else {
        if (k * m <= 331776) {
          if (k * n <= 602112) {
            return LAUNCH(4, 2, 4, 1, 128, 1);
          } else {
            if ((float)n / k <= 147) {
              return LAUNCH(2, 4, 2, 8, 32, 1);
            } else {
              return LAUNCH(4, 2, 4, 1, 128, 1);
            }
          }
        } else {
          if (k * n <= 2709504) {
            return LAUNCH(4, 2, 2, 1, 64, 1);
          } else {
            return LAUNCH(2, 4, 2, 8, 32, 1);
          }
        }
      }
    }
  }
#undef LAUNCH
}

}  // namespace

// Launch the matrix multiply kernel for the passed parameters.
template <typename T, bool TransposeLHS, bool TransposeRHS>
SNNStatus launch(BaseMemObject<T const>& lhs, BaseMemObject<T const>& rhs,
                 BaseMemObject<T>& output, int batches, int m, int k, int n,
                 T beta, cl::sycl::queue& queue) {
  auto device_name =
      queue.get_device().get_info<cl::sycl::info::device::name>();
  if (device_name.find("Fiji") != std::string::npos) {
    return launch_for_amd<T, TransposeLHS, TransposeRHS>(
        lhs, rhs, output, batches, m, k, n, beta, queue);
  }
  if (device_name.find("Intel(R) Gen9 HD Graphics NEO") != std::string::npos) {
    return launch_for_intelgpu<T, TransposeLHS, TransposeRHS>(
        lhs, rhs, output, batches, m, k, n, beta, queue);
  }
  if (device_name.find("Intel(R) Core(TM) i7-6700K CPU @ 4.00GHz") !=
      std::string::npos) {
    return launch_for_intelcpu<T, TransposeLHS, TransposeRHS>(
        lhs, rhs, output, batches, m, k, n, beta, queue);
  }
  if (device_name.find("Mali-G71") != std::string::npos) {
    return launch_for_arm<T, TransposeLHS, TransposeRHS>(
        lhs, rhs, output, batches, m, k, n, beta, queue);
  }
  return launch_with_tiles<T, TransposeLHS, TransposeRHS, 4, 4, 4>(
      lhs, rhs, output, batches, m, k, n, beta, queue, 8, 4, 1);
}

#define INSTANTIATE_LAUNCHER(DTYPE, TLHS, TRHS)                                \
  template SNN_EXPORT SNNStatus launch<DTYPE, TLHS, TRHS>(                     \
      BaseMemObject<DTYPE const> & input, BaseMemObject<DTYPE const> & filter, \
      BaseMemObject<DTYPE> & output, int batches, int m, int k, int n,         \
      DTYPE beta, cl::sycl::queue& queue);

#define INSTANTIATE_FOR_TYPE(DTYPE)        \
  INSTANTIATE_LAUNCHER(DTYPE, true, true)  \
  INSTANTIATE_LAUNCHER(DTYPE, false, true) \
  INSTANTIATE_LAUNCHER(DTYPE, true, false) \
  INSTANTIATE_LAUNCHER(DTYPE, false, false)

INSTANTIATE_FOR_TYPE(float);

#ifdef SNN_USE_DOUBLE
INSTANTIATE_FOR_TYPE(double);
#endif  // SNN_USE_DOUBLE

#ifdef SNN_USE_HALF
INSTANTIATE_FOR_TYPE(cl::sycl::half);
#endif  // SNN_USE_HALF

#undef INSTANTIATE_FOR_TYPE
#undef INSTANTIATE_LAUNCHER

}  // namespace internal
}  // namespace matmul
}  // namespace sycldnn

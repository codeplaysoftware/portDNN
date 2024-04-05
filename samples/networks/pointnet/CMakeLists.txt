cmake_minimum_required(VERSION 3.10.2)
set(_cxx_opts)
set(backend_providers)

if (SNN_TEST_PORTBLAS)
  include(HandlePortBLAS)
  list(APPEND _cxx_opts -DSNN_TEST_PORTBLAS=1)
  list(APPEND backend_providers PORTBLAS::portblas)
endif()

snn_executable(
  WITH_SYCL
  TARGET
    pointnet
  KERNEL_SOURCES
    pointnet.cc
  SOURCES
    pointnet.cc
  PUBLIC_LIBRARIES
    sycl_dnn
    ${backend_providers}
  CXX_OPTS
    ${_cxx_opts}
)

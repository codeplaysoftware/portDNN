FROM ubuntu:artful

# Default values for the build
ARG git_branch
ARG git_slug
ARG c_compiler
ARG cxx_compiler
ARG impl
ARG target

RUN apt-get -yq update

# Utilities
RUN apt-get install -yq --allow-downgrades --allow-remove-essential            \
    --allow-change-held-packages git wget apt-utils cmake unzip                \
    libboost-all-dev software-properties-common python-software-properties

RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test

RUN apt-get -yq update

# Clang 5.0
RUN if [ "${c_compiler}" = 'clang-5.0' ]; then apt-get install -yq             \
    --allow-downgrades --allow-remove-essential --allow-change-held-packages   \
     clang-5.0 libomp-dev; fi

# GCC 7
RUN if [ "${c_compiler}" = 'gcc-7' ]; then apt-get install -yq                 \
    --allow-downgrades --allow-remove-essential --allow-change-held-packages   \
    g++-7 gcc-7; fi

# OpenCL ICD Loader
RUN apt-get install -yq --allow-downgrades --allow-remove-essential           \
    --allow-change-held-packages ocl-icd-opencl-dev ocl-icd-dev opencl-headers

# Install build tools
RUN apt-get install -yq --allow-downgrades --allow-remove-essential           \
    --allow-change-held-packages mercurial doxygen graphviz 

RUN git clone https://github.com/${git_slug}.git -b ${git_branch} /SYCL-DNN

# Intel OpenCL Runtime
RUN if [ "${target}" = 'opencl' ]; then bash /SYCL-DNN/.travis/install_intel_opencl.sh; fi

# SYCL
RUN if [ "${impl}" = 'COMPUTECPP' ]; then cd /SYCL-DNN && bash /SYCL-DNN/.travis/build_computecpp.sh; fi

ENV CC=${c_compiler}
ENV CXX=${cxx_compiler}
ENV SYCL_IMPL=${impl}
ENV TARGET=${target}

CMD cd /SYCL-DNN && \
    if [ "${SYCL_IMPL}" = 'COMPUTECPP' ]; then \
      if [ "${TARGET}" = 'host' ]; then \
        COMPUTECPP_TARGET="host" CMAKE_ARGS="-DSNN_BUILD_DOCUMENTATION=OFF " ./build.sh /tmp/ComputeCpp; \
      else \
        /tmp/ComputeCpp/bin/computecpp_info && \
        COMPUTECPP_TARGET="intel:cpu" CMAKE_ARGS="-DSNN_BUILD_DOCUMENTATION=OFF " ./build.sh /tmp/ComputeCpp; \
      fi \
    else \
      echo "Unknown SYCL implementation ${SYCL_IMPL}"; return 1; \
    fi

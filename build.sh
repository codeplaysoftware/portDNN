#! /bin/bash

function display_help() {

cat <<EOT

To use build.sh to compile SYCL-DNN with ComputeCpp:

  ./build.sh "path/to/ComputeCpp"
  (the path to ComputeCpp can be relative)

  For example:
  ./build.sh /home/user/ComputeCpp

EOT
}


# Useless to go on when an error occurs
set -o errexit

# Minimal emergency case to display the help message whatever happens
trap display_help ERR

echo "build.sh entering mode: ComputeCpp"
CMAKE_ARGS="$CMAKE_ARGS -DCOMPUTECPP_PACKAGE_ROOT_DIR=$(readlink -f $1)"
CMAKE_ARGS="$CMAKE_ARGS -DSNN_TEST_EIGEN=OFF"
shift

NPROC=$(nproc)

function configure  {
  mkdir -p build && pushd build
  cmake .. $CMAKE_ARGS 
  popd
}

function mak  {
  pushd build && make -j$NPROC
  popd
}

function tst {
  pushd build
  ctest -VV --timeout 600
  popd
}

function main {
  configure
  mak
  tst
}

main

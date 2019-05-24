#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

VERSION=${VERSION:-""}

git ls-files | grep -E "*\.h$|*\.hpp$|*\.cc$|*\.cpp$|*\.cc.in$" | \
  xargs --max-procs=`nproc` --max-args=1 clang-format$VERSION -style=file -i

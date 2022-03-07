#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

git ls-files | grep -E "*.py$" | \
  xargs --max-procs=`nproc` --max-args=1 autopep8 -i -a -a -r

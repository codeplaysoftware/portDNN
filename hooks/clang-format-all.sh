#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

VERSION=''

git ls-files | grep -E "*\.h$|*\.hpp$|*\.cc$|*\.cpp$" | \
  xargs clang-format$VERSION -style=file -i

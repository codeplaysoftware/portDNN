#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

hooks/clang-format-all.sh
git diff --quiet

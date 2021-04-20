#!/usr/bin/env bash

set -eux

mkdir -p build
cd build
cmake -GNinja \
      -DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')" \
      ..
cmake --build .

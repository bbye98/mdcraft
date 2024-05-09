#!/bin/bash

cd "$(dirname "$0")/.." && \
python3 -m build && \
conda-build .
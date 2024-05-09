#!/bin/bash

cd "$(dirname "$0")/.." && \
ruff check && \
coverage run -m pytest && \
coverage html --omit="*/test*,*/__init__.py" -d tests/htmlcov && \
sphinx-build -b html docs/source docs -j auto
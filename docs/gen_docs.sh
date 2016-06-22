#!/bin/bash
set -euo pipefail
sphinx-apidoc -f -o api ../minpy
pushd docs > /dev/null
make html

#!/bin/bash
set -euo pipefail
sphinx-apidoc -f -M -H MinPy -A DMLC -o api ../minpy
make html

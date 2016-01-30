#!/usr/bin/env bash
sphinx-apidoc -f -o docs python
pushd docs > /dev/null
make html

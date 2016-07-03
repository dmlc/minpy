#!/usr/bin/env bash
set -euo pipefail
filepath=$(git rev-parse --show-toplevel)
pushd "${filepath}/.git" > /dev/null
rm -rf hooks
ln -s ../hooks .

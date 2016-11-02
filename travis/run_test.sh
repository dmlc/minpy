#!/bin/bash

if [ ${TASK} == "lint" ]; then
    python ./dmlc-core/scripts/lint.py python minpy
fi

#TODO : uncomment after adding mxnet compilation
#if [ ${TASK} == "example_test" ]; then
    # python examples/nn/cnn.py
#fi

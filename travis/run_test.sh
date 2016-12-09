#!/bin/bash

if [ ${TASK} == "lint" ]; then
    if [ $TRAVIS_PYTHON_VERSION == 2.7 ]; then
        python ./dmlc-core/scripts/lint.py minpy python minpy || exit -1
        exit 0
    else
        python ./dmlc-core/scripts/lint3.py minpy python minpy || exit -1
        exit 0
    fi
fi

if [ ${TASK} == "example_test" ]; then
    cd mxnet
    make all || exit -1
    cd python && python setup.py install
    cd ../..
    nosetests tests/unittest || exit -1
    exit 0
fi

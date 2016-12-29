#!/bin/bash

if [ ${TASK} == "lint" ]; then
    python ./dmlc-core/scripts/lint.py minpy python minpy || exit -1
    exit 0
fi

if [ ${TASK} == "example_test" ]; then
    cd mxnet
    make all || exit -1
    cd python && python setup.py install
    cd ../..
    nosetests -v tests/unittest || exit -1
    # -s allows stdout. Avoids travis killing test for 10-min no output.
    nosetests -vs tests/perm_test || exit -1
    exit 0
fi

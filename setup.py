#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import setuptools
import os.path

with open(os.path.join(os.path.dirname(__file__), 'VERSION')) as f:
    version = f.readline().strip()

setuptools.setup(
    name='minpy',
    version=version,
    description='Pure NumPy practice with third-party operator integration.',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'mxnet',
        'semantic_version',
    ],
    url='https://github.com/dmlc/minpy')

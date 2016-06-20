#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import setuptools

setuptools.setup(
    name='minpy',
    version='0.0.3',
    description='Pure NumPy practice with third-party operator integration.',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'mxnet',
    ],
    url='https://github.com/dmlc/minpy')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import setuptools

setuptools.setup(name='minpy',
                 version='0.0.1',
                 packages=setuptools.find_packages(),
                 install_requires=[
                     'enum',
                     'numpy',
                     'mxnet',
                 ],
                 url='https://github.com/dmlc/minpy')

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import setuptools
import os.path

with open(os.path.join(os.path.dirname(__file__), 'VERSION')) as f:
    version = f.readline().strip()

setuptools.setup(
    name='minpy',
    version=version,
    description='Pure NumPy practice with third-party operator integration.',
    maintainer='DMLC',
    maintainer_email='minerva-support@googlegroups.com',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pillow',
        'mxnet',
        'semantic-version',
    ],
    data_files=[('', ['VERSION'])],
    url='https://github.com/dmlc/minpy')

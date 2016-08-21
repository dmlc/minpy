#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Mock numpy random module """
#pylint: disable= invalid-name
from __future__ import absolute_import

import sys
from minpy.numpy.mocking import Module

_old = {
    '__name__' : __name__,
}

sys.modules[__name__] = Module(_old, 'random')

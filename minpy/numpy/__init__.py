#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Mock numpy package """
#pylint: disable= invalid-name
import sys

from minpy.array import Value
from minpy.numpy.mocking import Module

from . import random

_old = {
    '__path__' : __path__,
    '__name__' : __name__,
    'random' : random,
}

_mod = Module(_old)
Value._ns = _mod
sys.modules[__name__] = _mod

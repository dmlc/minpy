#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Mock numpy package """
#pylint: disable= invalid-name
import sys
import numpy

from minpy.array import Value
from minpy.numpy.mocking import Module, NameInjector

from . import random

_old = {
    '__path__' : __path__,
    '__name__' : __name__,
    'random' : random,
}

injected_name = {'float', 'float16', 'float32', 'float64', 'int', 'int8',
                 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32',
                 'uint64', 'bool',
                 'newaxis',
                 'e', 'pi', 'inf'}
_mod = Module(_old, name_injector=NameInjector(injected_name, numpy))
Value._ns = _mod
sys.modules[__name__] = _mod

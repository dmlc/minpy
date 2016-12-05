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
    '__path__': __path__,
    '__name__': __name__,
    'random': random,
}

_mod = Module(
    _old,
    name_injector=NameInjector(
        numpy, name='numpy injector', injected_type={float, int, type(None), type}))
Value._ns = _mod # pylint: disable= protected-access
sys.modules[__name__] = _mod

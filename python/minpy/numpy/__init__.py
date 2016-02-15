#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import sys
from .mocking import Module
import numpy
from .. import array

_old = {
    '__path__' : __path__,
    '__name__' : __name__,
}

_mod = Module(_old)
array.Array._ns = _mod
sys.modules[__name__] = _mod

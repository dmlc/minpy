#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Mock numpy package """
#pylint: disable= invalid-name
from __future__ import absolute_import

import sys
from .mocking import Module
from .. import array

from . import random

_old = {
    '__path__' : __path__,
    '__name__' : __name__,
    'random' : random,
}

_mod = Module(_old)
array.Value._ns = _mod
sys.modules[__name__] = _mod

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import sys
from .mocking import Module
import numpy
from .. import array

_mod = Module(numpy.__dict__)
array.Array._ns = _mod
sys.modules[__name__] = _mod

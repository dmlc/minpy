#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import sys
from .mocking import Module

_old = {
    '__name__' : __name__,
}

sys.modules[__name__] = Module(_old, 'random')

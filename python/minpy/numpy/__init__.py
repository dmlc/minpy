#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
import sys


class Module(object):
    """Module level dynamic lookup."""

    def __getattr__(self, name):
        pass

sys.modules[__name__] = Module()

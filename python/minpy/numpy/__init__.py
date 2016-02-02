#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import sys
from .mocking import Module

sys.modules[__name__] = Module(globals())

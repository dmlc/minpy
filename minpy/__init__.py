#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable= invalid-name
"""minpy root module"""
from __future__ import absolute_import

import mxnet as mx

from .dispatch import policy
from .dispatch.policy import PreferMXNetPolicy

# Global config
Config = {'modules': [], }
Config['default_policy'] = PreferMXNetPolicy()

# Import minpy.numpy package to do some initialization.
from . import numpy  # pylint: disable= wrong-import-position

def set_global_policy(plc):
    """ Set global policy for all modules. This will also change default policy
    in future imported modules.

    Parameters
    ----------
    plc : str or Policy object
        The policy to set.
    """
    Config['default_policy'] = policy.create(plc)
    for mod in Config['modules']:
        mod.generate_attrs(Config['default_policy'])

def get_global_policy():
    """Return the current global policy."""
    return Config['default_policy']

def check_mxnet_version():
    """Check whether MXNet version satisfies minimum requirement."""
    supported = (0, 9, 2)
    current = tuple(int(v) for v in mx.__version__.split("."))
    succ = True
    for v1, v2 in zip(current, supported):
        if v1 > v2:
            succ = True
            break
        elif v1 < v2:
            succ = False
            break
    assert succ, "Unsupported MXNet version: %s; minimum version required: 0.9.2" % mx.__version__

check_mxnet_version()

wrap_policy = policy.wrap_policy

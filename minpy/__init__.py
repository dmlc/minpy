#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable= invalid-name
"""minpy root module"""
from __future__ import absolute_import

from .dispatch import policy
from .dispatch.policy import PreferMXNetPolicy

# Global config
Config = {'modules': [], }
Config['default_policy'] = PreferMXNetPolicy()

def set_global_policy(policy):
    """ Set global policy for all modules. This will also change default policy
    in future imported modules.

    :param minpy.dispatch.policy.Policy policy: New policy.
    """
    Config['default_policy'] = policy.create(policy)
    for mod in Config['modules']:
        mod.generate_attrs(Config['default_policy'])

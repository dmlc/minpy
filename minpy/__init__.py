#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable= invalid-name
"""minpy root module"""
from __future__ import absolute_import
from minpy.dispatch.policy import PreferMXNetPolicy, OnlyNumPyPolicy, OnlyMXNetPolicy, wrap_policy
from minpy.dispatch.policy import AutoBlacklistPolicy
# Global config
Config = {'modules': [], }
Config['default_policy'] = PreferMXNetPolicy()


def set_global_policy(policy):
    """ Set global policy for all modules. This will also change default policy
    in future imported modules.

    :param minpy.dispatch.policy.Policy policy: New policy.
    """
    for mod in Config['modules']:
        mod.set_policy(policy)
    Config['default_policy'] = policy

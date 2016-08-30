#!/usr/bin/env python
# -*- coding: utf-8 -*-
#pylint: disable= invalid-name
from __future__ import absolute_import
from minpy.dispatch.policy import PreferMXNetPolicy, OnlyNumPyPolicy, OnlyMXNetPolicy
# Global config
Config = {'modules': [], }
Config['default_policy'] = PreferMXNetPolicy()


def set_global_policy(policy):
    """ Set global policy for all imported modules
    
    This function only affects imported modules.

    :param minpy.dispatch.policy.Policy policy: New policy.
    """
    for mod in Config['modules']:
        mod.set_policy(policy)


def set_default_policy(policy):
    """ Set default policy for future imported modules

    This function only affects modules which haven't been imported.

    :param minpy.dispatch.policy.Policy policy: New policy.
    """
    Config['default_policy'] = policy

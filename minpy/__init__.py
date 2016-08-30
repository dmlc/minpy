#!/usr/bin/env python
# -*- coding: utf-8 -*-
#pylint: disable= invalid-name

# Global config
Config = {'modules': [], }


def set_global_policy(policy):
    """ Set global policy for all imported modules

    :param minpy.dispatch.policy.Policy policy: New policy.
    """
    for mod in Config['modules']:
        mod.set_policy(policy)

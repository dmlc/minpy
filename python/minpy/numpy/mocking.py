#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import

import sys
import importlib

from ..utils import log
from ..dispatch import registry
from ..dispatch import policy
from ..array_variants import * # import all array_variants names
from .. import array

class DynamicLookupError(KeyError):
    pass

class Module(object):
    """Module level dynamic lookup."""

    def __init__(self, old, name=None):
        self._registry = registry.Registry()
        self._policy = policy.PreferMXNetPolicy()
        self._logger = log.get_logger(name)
        self._logger.info('Initialize module: {}'.format(old['__name__']))
        self._old = old
        for vname in variants:
            if name == None:
                modname = 'minpy.array_variants.{}'.format(vname)
            else:
                modname = 'minpy.array_variants.{}.{}'.format(vname, name)
            mod = importlib.import_module(modname)
            #TODO better wrapper?
            def primitive_wrapper(func):
                return array.Primitive(func, variants[vname][1])
            # register all primitives of the module
            mod.register_primitives(self._registry, primitive_wrapper)
            def primitive_getter(name):
                return self._registry.get(name, variants[vname][1])
            # define gradients of primitives
            mod.def_grads(self._registry, primitive_getter)
        self._logger.info('Import {} primitives'.format(len(self._registry._reg)))
    
    def __getattr__(self, name):
        self._logger.info('Look up name {}'.format(name))
        # Special members for internal use.
        if name == '__registry__':
            return self._registry
        elif name in self._old:
            return self._old[name]
        elif self._registry.has_name(name):
            return policy.resolve_name(name, self._registry, self._policy)
        else:
            raise DynamicLookupError()

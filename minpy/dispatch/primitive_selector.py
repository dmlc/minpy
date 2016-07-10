#!/usr/bin/env python
# -*- coding: utf-8 -*-
#pylint: disable= invalid-name
"""Lazy evaluation of primitive selection."""
from __future__ import absolute_import
from __future__ import print_function

from minpy.utils import log
from minpy.dispatch import policy

_logger = log.get_logger(__name__)


class PrimitiveSelector(object):
    """Primitive selector class that behaves like normal function but instead pass all the
    arguments to the policy to choose appropriate primitive.
    """
    __slots__ = ['_name', '_registry', '_policy']

    def __init__(self, name, reg, plc):
        self._name = name
        self._registry = reg
        self._policy = plc

    @property
    def name(self):
        """Get the name of this function.

        :return: Name of function.
        """
        return self._name

    def __call__(self, *args, **kwargs):
        """Call policy to choose the real primitive and then call the returned function with
        the given arguments.
        """
        prim = policy.resolve_name(self._name, self._registry, self._policy,
                                   args, kwargs)
        _logger.debug('Found primitive "{}" with type {}.'.format(
            self._name, prim.typestr))
        return prim(*args, **kwargs)

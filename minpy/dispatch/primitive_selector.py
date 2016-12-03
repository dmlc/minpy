#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Lazy evaluation of primitive selection."""
from __future__ import absolute_import
from __future__ import print_function


class PrimitiveSelector(object):
    """Primitive selector class that behaves like normal function but instead pass all the
    arguments to the policy to choose appropriate primitive call.
    """
    __slots__ = ['_name', '_mod']

    def __init__(self, name, mod):
        self._name = name
        self._mod = mod

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
        return self._mod.policy.resolve_call(
            self._name, self._mod.__registry__, args, kwargs)

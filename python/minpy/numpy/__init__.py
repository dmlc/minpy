#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
import sys
import typing
from ..utils import log
from ..dispatch import registry

_logger = log.get_logger(__name__)

_old_definitions = {
    '__name__': __name__,
    '__loader__': __loader__,
    '__package__': __package__,
    '__spec__': __spec__,
    '__file__': __file__,
}

try:
    _old_definitions['__cached__'] = __cached__
except NameError:
    pass


class DynamicLookupError(KeyError):
    pass


class Module(object):
    """Module level dynamic lookup."""

    _registry = registry.Registry()

    def __getattr__(self, name: str) -> typing.Any:
        _logger.info('Look up name {}'.format(name))
        # Special members for internal use.
        if name == '__registry__':
            return self._registry
        elif name in _old_definitions:
            return _old_definitions[name]
        elif self._registry.has_name(name):
            return None  # TODO policy here
        else:
            raise DynamicLookupError()

# TODO initialize registry here by importing "arary variants"
sys.modules[__name__] = Module()

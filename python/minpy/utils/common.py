#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Common utility functions."""
import functools


def assert_type(t, method=False):
    """Assert argument types.

    Args:
        t: Type expected.
        method: Whether it is a method, skip first argument `self`.

    Returns:
        A decorator.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            for arg in (args[1:] if method else args):
                assert(type(arg) == t)
            for _, kwarg in kwargs.items():
                assert(type(kwarg) == t)
            return func(*args, **kwargs)
        return wrapped
    return decorator

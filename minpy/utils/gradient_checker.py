#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable= invalid-name
"""Gradient Checker"""
from __future__ import absolute_import

import numpy as np
from minpy.core import grad
from minpy.array import Value

def quick_grad_check(fun,
                     arg,
                     verbose=True,
                     eps=1e-2,
                     rtol=1e-2,
                     atol=1e-2,
                     rs=None):
    # pylint: disable= too-many-arguments
    """
    Checks the gradient of a function (w.r.t. to its first arg) in a random direction

    Args:
        fun:
            The function for gradient checking
        arg:
            Gradient checking point
        verbose:
            Whether print some debug information
        eps:
            Epsilon in computing numerical gradient
        rtol:
            Relative tolerance
        atol:
            Absolute tolerance
        rs:
            RandomState used in generating random direction
    """
    if rs is None:
        rs = np.random.RandomState()

    if isinstance(arg, Value): # convert it to numpy value
        arg = arg.asnumpy()

    random_dir = rs.standard_normal(np.shape(arg))
    random_dir = random_dir / np.sqrt(np.sum(random_dir * random_dir))

    grad_fun = grad(fun)
    unary_fun = lambda x: fun(arg + x * random_dir).asnumpy()
    numeric_grad = (unary_fun(eps / 2) - unary_fun(-eps / 2)) / eps
    analytic_grad = np.sum(grad_fun(arg).asnumpy() * random_dir)

    passed = np.allclose(numeric_grad, analytic_grad, rtol=rtol, atol=atol)

    if verbose:
        if passed:
            print("Gradient projection OK (numeric grad: {0}, analytic grad: {1})".format(\
                numeric_grad, analytic_grad))
        else:
            print("Check failed! numeric={0}, analytic={1}".format(\
                numeric_grad, analytic_grad))

    return passed

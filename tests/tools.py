#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import numpy.testing as npt
import numpy as np

xs = np.linspace(-1, 1, 1000)

def assert_close(c1, c2, xx=xs, *args, **kwargs):
    """
    Check that two callable objects are close approximations of one another
    by evaluating at a number of points on an interval (default [-1,1]).
    """
    npt.assert_allclose(c1(xx), c2(xx), *args, **kwargs)


#------------------------------------------------------------------------------
# Functions utilised in the unit-tests
#------------------------------------------------------------------------------

ufunc_list = [np.arccos, np.arcsin, np.arcsinh, np.arctan, np.arctanh, np.cos, np.sin, np.tan, np.cosh, np.sinh, np.tanh, np.exp, np.exp2, np.expm1, np.log, np.log2, np.log1p, np.sqrt, np.ceil, np.trunc, np.fabs, np.floor, np.abs]

def name_func(ufunc):
    return ufunc.__name__

def map_ab_ui(x, a, b):
    return (2.0*x-a-b)/(b-a)

def map_ui_ab(t, a, b):
    return 0.5*(b-a)*t + 0.5*(a+b)

def Identity(x):
    return x

def One(x):
    return np.ones_like(x, dtype=float)

def Zero(x):
    return np.zeros_like(x, dtype=float)

def circle(x, period=2):
    return np.array([np.cos(2*np.pi/period*x), np.sin(2*np.pi/period*x)],).T

def f(x):
    return np.sin(6*x) + np.sin(30*np.exp(x))

def fd(x):
    """
    Derivative of f
    """
    return 6*np.cos(6*x) + np.cos(30*np.exp(x))*30*np.exp(x)


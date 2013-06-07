#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import numpy as np

def circle(x):
    return np.array([np.cos(np.pi*x), np.sin(np.pi*x)],).T

def f(x):
    return np.sin(6*x) + np.sin(30*np.exp(x))

def fd(x):
    """
    Derivative of f
    """
    return 6*np.cos(6*x) + np.cos(30*np.exp(x))*30*np.exp(x)

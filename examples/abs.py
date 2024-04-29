#!/usr/bin/env python
"""
This example shows the limits of Chebfun without splitting.
The function abs may be represented with four interpolation points as a piecewise Chebfun.
Instead, the dichotomy algorithm on [-1,1] produces more and more points, and does not even converge.
"""
from pychebfun import Chebfun, compare
import numpy as np
import functools


np.seterr(all='raise')

def abse(x, epsilon):
    """
    Smooth approximation of the abs function.
    """
    return np.sqrt(epsilon + x**2)


abses = [functools.partial(abse, epsilon=pow(10,-k)) for k in range(4)]
cs = []
for f in abses:
    try:
        c = Chebfun.from_function(f)
    except c.NoConvergence:
        break
    else:
        cs.append(c)

c,f = list(zip(cs,abses))[3]
compare(c, f)

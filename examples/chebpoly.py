#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

from pychebfun import *


def plot_first_chebyshev(N, alpha_coeff=10):
    """
    Plot the N first Chebyshev polynomials.

    alpha_coeff:  how slowly higher order polynomial fade away in the plot
    """
    cs = [Chebfun.basis(deg) for deg in range(N)]
    for k,c in enumerate(cs):
        plot(c, with_interpolation_points=False, color='black', alpha=alpha_coeff/(k+alpha_coeff))
    return cs

plot_first_chebyshev(50)

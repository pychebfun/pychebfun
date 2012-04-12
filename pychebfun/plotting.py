"""
Chrbfun plotting tools

CREATED BY:
----------

    -- Chris Swierczewski <cswiercz@gmail.com>

"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

from .chebfun import Chebfun

def chebpolyplot(f, Nmax=100, normpts=1000, ord=2, compare=False, points_only=False):
    """
    Plots the number of Chebyshev points vs. norm accuracy of the
    chebfun interpolant.

    INPUTS:

        -- f: Python, Numpy, or Sage function
        
        -- Nmax: (default = 100) maximum number of interpolating points
        
        -- normpts: (default = 1000) number of sample points to take
                    when computing the norm

        -- ord: (default = None) {1,-1,2,-2,inf,-inf,'fro'} order of the norm

        -- compare: (default = False) compare normal with Chebyshev interpol

        -- points_only: (default = False) return only the plot points. do
                        not actually plot the graph

    OUTPUTS:

        -- Nvals: array of interpolating point numbers

        -- normvalscheb: array of norm values from Chebyshev interpolation

        -- normvalsequi: array of norm values from equidistant interpolation
    """

    x            = np.linspace(-1,1,normpts)    
    Nvals        = range(10,Nmax,10)
    normvalscheb = [la.norm(f(x)-Chebfun(f,N=n)(x), ord=ord) for n in Nvals]

    if compare:
        normvalsequi = [la.norm(f(x)-Chebfun(f,N=n,spacing='equidistant')(x), ord=ord) for n in Nvals]

    # plot this 
    if not points_only:
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.plot(Nvals,np.log10(normvalscheb),'r', label='Chebyshev Interpolation')
        ax.plot(Nvals,np.log10(normvalscheb),'r.', markersize=10)
        if compare:
            ax.plot(Nvals,np.log10(normvalsequi),'k', label='Equispaced Interpolation')
            ax.plot(Nvals,np.log10(normvalsequi),'k.', markersize=10)
        ax.set_xlabel('Number of Interpolating Points')
        ax.set_ylabel('%s-norm Error ($\log_{10}$-scale)' %(str(ord)))
        ax.legend(loc='best')

    return ax




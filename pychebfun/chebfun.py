#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

"""
Chebfun module
==============

.. moduleauthor :: Chris Swierczewski <cswiercz@gmail.com>
.. moduleauthor :: Olivier Verdier <olivier.verdier@gmail.com>
.. moduleauthor :: Gregory Potter <ghpotter@gmail.com>


"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import sys
from functools import wraps

from scipy.interpolate import BarycentricInterpolator as Bary
from scipy.fftpack     import dct

def cast_scalar(method):
    """
    Used to cast scalar to Chebfuns
    """
    @wraps(method)
    def new_method(self, other):
        if np.isscalar(other):
            other = Chebfun([other])
        return method(self, other)
    return new_method

emach     = sys.float_info.epsilon                        # machine epsilon

class Chebfun(object):
    """
    Construct a Lagrange interpolating polynomial over the Chebyshev points.

    """
    max_nb_dichotomy = 12 # maximum number of dichotomy of the interval

    class NoConvergence(Exception):
        """
        Raised when dichotomy does not converge.
        """

    def __init__(self, f=None, N=0, chebcoeff=None,):
        """
Create a Chebyshev polynomial approximation of the function $f$ on the interval :math:`[-1,1]`.

:param callable f: Python, Numpy, or Sage function
:param int N: (default = None)  specify number of interpolating points
:param np.array chebcoeff: (default = np.array(0)) specify the coefficients of a Chebfun
        """

        if self.record:
            self.intermediate = []
            self.bnds = []

        if np.isscalar(f):
            f = [f]

        try:
            i = iter(f) # interpolation values provided
        except TypeError:
            pass
        else:
            vals = np.array(f)
            N = len(vals)-1
            if N:
                self.ai = self.chebpolyfit(vals,N, sample=False)
                self.x = self.interpolation_points(N)
            else: # just one value provided
                self.ai = vals.copy()
                self.x = [1.]
            self.f = vals.copy()
            self.p  = Bary(self.x, self.f)
            return None

        if isinstance(f, Chebfun): # copy if f is another Chebfun
            self.ai = f.ai.copy()
            self.x = f.x
            self.f = f.f

        if chebcoeff is not None: # if the coefficients of a Chebfun are given

            self.N = N = len(chebcoeff)
            self.ai = chebcoeff
            self.f = self.idct(chebcoeff)
            self.x = self.interpolation_points(N-1)
            self.p = Bary(self.x, self.f)

        else: # if the coefficients of a Chebfun are not given
            if not N: # N is not provided
                # Find out the right number of coefficients to keep
                for k in xrange(2,self.max_nb_dichotomy):
                    N = pow(2,k)

                    coeffs = self.chebpolyfit(f,N, sample=True)

                    # 3) Check for negligible coefficients
                    #    If within bound: get negligible coeffs and bread
                    bnd = 128*emach*abs(np.max(coeffs))
                    if self.record:
                        self.bnds.append(bnd)
                        self.intermediate.append(coeffs)

                    last = abs(coeffs[-2:])
                    if np.all(last <= bnd):
                        break
                else:
                    raise self.NoConvergence(last, bnd)


                # End of convergence loop: construct polynomial
                [inds]  = np.where(abs(coeffs) >= bnd)
                N = inds[-1]

                if self.record:
                    self.bnds.append(bnd)
                    self.intermediate.append(coeffs)
            else:
                nextpow2 = int(np.log2(N))+1
                coeffs = self.chebpolyfit(f,pow(2,nextpow2), sample=True)

            self.ai = coeffs[:N+1]
            self.x  = self.interpolation_points(N)
            self.f  = f(self.x)
            self.p  = Bary(self.x, self.f)
            


    record = False # whether to record convergence information

    def even_data(self, data):
        """
        Construct Extended Data Vector (equivalent to creating an
        even extension of the original function)
        """
        return np.hstack([data,data[-2:0:-1]])

    def interpolation_points(self, N):
        """
        N+1 Chebyshev points in [-1,1], boundaries included
        """
        return np.cos(np.arange(N+1)*np.pi/N)

    def sample(self, f, N):
        x = self.interpolation_points(N)
        return f(x)

    def dct(self, data):
        """
        Compute DCT
        """
        N = len(data)//2
        dctdata     = dct(data[:N+1], 1)
        dctdata     /= N
        dctdata[0]  /= 2.
        dctdata[-1] /= 2.
        return dctdata

    def chebpolyfit(self, f, N, sample=True):
        """
        Compute Chebyshev coefficients of a function f on N points.
        """
        if sample:
            sampled = self.sample(f,N)
        else: # f is a vector
            sampled = f
        evened = self.even_data(sampled)
        coeffs = self.dct(evened)
        return coeffs

    def __repr__(self):
        return "<Chebfun({0})>".format(len(self))

    def idct(self, chebcoeff):
        """
        Compute the inverse DCT
        """
        N = len(chebcoeff)

        data = 2.*chebcoeff
        data[0] *= 2
        data[-1] *= 2
        data *= N

        idctdata = dct(data,1)/(4*N)
        return idctdata


    #
    # Basic Operator Overloads
    #
    def __call__(self, x):
        return self.p(x)

    def __len__(self):
        return self.p.n

    def __nonzero__(self):
        """
        Test for difference from zero (up to tolerance)
        """
        return not np.allclose(self.chebyshev_coefficients(),0)

    def __eq__(self, other):
        return not(self - other)

    def __neq__(self, other):
        return not (self == other)

    @cast_scalar
    def __add__(self, other):
        """
        Addition
        """
        return Chebfun(lambda x: self(x) + other(x),)

    __radd__ = __add__


    @cast_scalar
    def __sub__(self, other):
        """
        Chebfun subtraction.
        """
        return Chebfun(lambda x: self(x) - other(x),)

    def __rsub__(self, other):
        return -(self - other)


    @cast_scalar
    def __mul__(self, other):
        """
        Chebfun multiplication.
        """
        return Chebfun(lambda x: self(x) * other(x),)

    __rmul__ = __mul__

    @cast_scalar
    def __div__(self, other):
        """
        Chebfun division
        """
        return Chebfun(lambda x: self(x) / other(x),)

    __truediv__ = __div__

    @cast_scalar
    def __rdiv__(self, other):
        return Chebfun(lambda x: other(x)/self(x))

    __rtruediv__ = __rdiv__

    def __neg__(self):
        """
        Chebfun negation.
        """
        return Chebfun(lambda x: -self(x),)

    def __pow__(self, other):
        return Chebfun(lambda x: self(x)**other)


    def sqrt(self):
        """
        Square root of Chebfun.
        """
        return Chebfun(lambda x: np.sqrt(self(x)),)

    def __abs__(self):
        """
        Absolute value of Chebfun. (Python)

        (Coerces to NumPy absolute value.)
        """
        return Chebfun(lambda x: np.abs(self(x)),)

    def abs(self):
        """
        Absolute value of Chebfun. (NumPy)
        """
        return self.__abs__()

    def sin(self):
        """
        Sine of Chebfun
        """
        return Chebfun(lambda x: np.sin(self(x)),)


    #
    # Numpy / Scipy Operator Overloads
    #

    def chebyshev_coefficients(self):
        return self.ai

    def integral(self):
        """
        Evaluate the integral of the Chebfun over the given interval using
        Clenshaw-Curtis quadrature.
        """
        ai2 = self.ai[::2]
        n = len(ai2)
        Tints = 2/(1-(2*np.arange(n))**2)
        val = np.sum(Tints*ai2)

        return val


    def integrate(self):
        """
        Return the Chebfun representing the integral of self over the domain.

        (Simply numerically integrates the underlying Barcentric polynomial.)
        """
        return Chebfun(self.p.integrate)

 
    def derivative(self):
        return self.differentiate()

    def differentiate(self):
        """
        Return the Chebfun representing the derivative of self. Uses spectral
        methods for accurately constructing the derivative.
        """
        # Compute new ai by doing a backsolve

        # If a_i and b_i are the kth Chebyshev polynomial expansion coefficient
        # Then b_{i-1} = b_{i+1} + 2ia_i; b_N = b_{N+1} = 0; b_0 = b_2/2 + a_1

        N = len(self.ai)

        bi = np.array([2.*(N-1)*self.ai[-2], 2.*N*self.ai[-1]])

        for i in np.arange(N-2, 1, -1):
            bi = np.append(bi[1] + 2.*i*self.ai[i], bi)
        bi = np.append(bi[1]/2. + self.ai[0], bi)

        return Chebfun(self, chebcoeff=bi)

    def roots(self):
        """
        Return the roots of the chebfun.
        """
        N            = len(self.ai)
        coeffs       = np.append(np.array([self.ai[N-k-1] for k in np.arange(N)]), self.ai[1:])
        coeffs[N-1] *= 2
        zNq          = np.poly1d(coeffs)
        return np.unique(np.array([np.real(r) for r in zNq.roots if abs(r) > 0.99999999 and abs(r) < 1.00000001]))
        
    plot_res = 1000

    def plot(self, interpolation_points=True, *args, **kwargs):
        xs = np.linspace(-1,1,self.plot_res)
        axis = plt.gca()
        axis.plot(xs,self(xs), *args, **kwargs)
        if interpolation_points:
            current_color = axis.lines[-1].get_color() # figure out current colour
            axis.plot(self.x, self.f, marker='.', linestyle='', color=current_color)
        plt.plot()

    def chebcoeffplot(self, *args, **kwds):
        """
        Plot the coefficients.
        """
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        
        data = np.log10(np.abs(self.ai))
        ax.plot(data, 'r' , *args, **kwds)
        ax.plot(data, 'r.', *args, **kwds)

        return ax

    def plot_interpolating_points(self):
        plt.plot(self.x, self.f)

    def compare(self,f,*args,**kwds):
        """
        Plots the original function against its chebfun interpolant.
        
        INPUTS:

            -- f: Python, Numpy, or Sage function
        """
        x   = np.linspace(-1,1,10000)
        fig = plt.figure()
        ax  = fig.add_subplot(211)
        
        ax.plot(x,f(x),'#dddddd',linewidth=10,label='Actual', *args, **kwds)
        self.plot(color='red', label='Chebfun Interpolant (d={0})'.format(len(self)), *args, **kwds)
        ax.legend(loc='best')

        ax  = fig.add_subplot(212)
        ax.plot(x,abs(f(x)-self(x)),'k')

        return ax

def chebpoly(n):
    if not n:
        return Chebfun(np.array([1.]))
    vals = np.ones(n+1)
    vals[-1::-2] = -1
    return Chebfun(vals)


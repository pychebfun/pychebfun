#!/usr/bin/env python
"""
Chebfun module
==============

.. moduleauthor :: Chris Swierczewski <cswiercz@gmail.com>
.. moduleauthor :: Olivier Verdier <olivier.verdier@gmail.com>
.. moduleauthor :: Gregory Potter <ghpotter@gmail.com>


"""
# coding: UTF-8
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

import sys
from functools import wraps

from scipy.interpolate import BarycentricInterpolator as Bary

def differentiator(A):
    """Differentiate a set of Chebyshev polynomial expansion 
       coefficients
       Originally from http://www.scientificpython.net/1/post/2012/04/chebyshev-differentiation.html
        + bug fixing
       """
    m, n = A.shape
    SA = A*np.outer(2*np.arange(m), np.ones(n))
    DA = np.zeros((m, n))
    if m == 1: # constant
        return np.zeros([1, n])
    if m == 2: # linear
        return A[1:2, :]
    DA[m-3:m-1, :] = SA[m-2:m, :]   
    for j in range(int(np.floor(m/2)-1)):
        k = m-3-2*j
        DA[k, :] = SA[k+1, :] + DA[k+2, :]
        DA[k-1, :] = SA[k, :] + DA[k+1, :]
    DA[0, :] = (SA[1, :] + DA[2, :])*0.5
    return DA

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

    def init_from_data(self, data):
        """
        The data provided are the values at the Chebyshev points
        """
        vals = np.array(data)
        N = len(vals)-1
        self.ai = chebpolyfit(vals)
        self.x = interpolation_points(N)
        self.f = vals.copy()
        self.p  = interpolate(self.x, self.f)

    def init_from_chebfun(self, other):
        """
        Initialise from another instance of Chebfun
        """
        self.ai = other.ai.copy()
        self.x = other.x
        self.f = other.f
        self.p = other.p

    def init_from_chebcoeff(self, chebcoeff):
        """
        Initialise from provided Chebyshev coefficients
        """
        ## if len(np.shape(chebcoeff)) == 1: # make sure the data is a matrix
        ## 	chebcoeff = np.reshape(chebcoeff, (-1, 1))

        N = len(chebcoeff)
        self.ai = chebcoeff
        self.f = idct(chebcoeff)
        self.x = interpolation_points(N-1)
        self.p = interpolate(self.x, self.f)

    def dichotomy(self, f, kmin=None, kmax=None, raise_no_convergence=True):
        """
        Compute the coefficients for a function f by dichotomy.
        kmin, kmax: log2 of number of interpolation points to try
        raise_no_convergence: whether to raise an exception if the dichotomy does not converge
        """
        if kmin is None:
            kmin = 2
        if kmax is None:
            kmax = self.max_nb_dichotomy

        if self.record:
            self.intermediate = []
            self.bnds = []

        for k in xrange(kmin, kmax):
            N = pow(2, k)

            sampled = sample_function(f, N)
            coeffs = chebpolyfit(sampled)

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
            if raise_no_convergence:
                raise self.NoConvergence(last, bnd)
        inds  = np.nonzero(abs(coeffs) >= bnd)
        Nmax = inds[0][-1]
        return coeffs, Nmax

    def init_from_function(self, f, N=None):
        """
        Initialise from a function to sample.
        N: optional parameter which indicates the range of the dichotomy
        """
        if N is not None: # N is provided
            nextpow2 = int(np.log2(N))+1
            kmin = nextpow2
            kmax = nextpow2+1
            raise_no_convergence = False
        else:
            kmin = None
            kmax = None
            raise_no_convergence = True

        # Find out the right number of coefficients to keep
        coeffs, Nmax = self.dichotomy(f, kmin, kmax, raise_no_convergence)


        self.ai = coeffs[:Nmax+1]
        self.x  = interpolation_points(Nmax)
        self.f  = f(self.x)
        self.p  = interpolate(self.x, self.f.T)

    def __init__(self, f=None, N=None, chebcoeff=None,):
        """
Create a Chebyshev polynomial approximation of the function $f$ on the interval :math:`[-1, 1]`.

:param callable f: Python, Numpy, or Sage function
:param int N: (default = None)  specify number of interpolating points
:param np.array chebcoeff: (default = np.array(0)) specify the coefficients of a Chebfun
        """

        if np.isscalar(f):
            f = [f]

        try:
            iter(f) # interpolation values provided
        except TypeError:
            pass
        else:
            self.init_from_data(f)
            return

        if isinstance(f, Chebfun): # copy if f is another Chebfun
            self.init_from_chebfun(f)
            return

        if chebcoeff is not None: # if the coefficients of a Chebfun are given
            self.init_from_chebcoeff(chebcoeff)
            return

        # from this point, we assume that f is a function
        if f is not None:
            self.init_from_function(f, N)

        # at this point, none of the initialisation worked, the Chebfun object is empy
        # but may be initialised manually with one of the init_ methods



    record = False # whether to record convergence information


    def __repr__(self):
        return "<Chebfun({0})>".format(len(self))

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
        return not np.allclose(self.chebyshev_coefficients(), 0)

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
        val = np.sum((Tints*ai2.T).T, axis=0)

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
        bi = differentiator(self.ai)
        return Chebfun(chebcoeff=bi)

    def roots(self):
        """
        Return the roots of the first component of the chebfun.
        """
        ai = self.ai[:, 0]
        N = len(ai)
        coeffs = np.hstack([ai[-1::-1], ai[1:]])
        coeffs[N-1] *= 2
        zNq = np.poly1d(coeffs)
        roots = np.array([np.real(r) for r in zNq.roots if np.allclose(abs(r), 1.)])
        return np.unique(roots)

    plot_res = 1000

    def plot(self, with_interpolation_points=True, *args, **kwargs):
        xs = np.linspace(-1, 1, self.plot_res)
        axis = plt.gca()
        ys = self(xs)
        # figuring out the dimension of the data; should be factored out
        shape = np.shape(ys)
        if len(shape) == 1:
            dim = 1
        else:
            dim = shape[1]
        if dim == 1:
            axis.plot(xs, ys, *args, **kwargs)
        elif dim == 2:
            axis.plot(ys[:, 0], ys[:, 1], *args, **kwargs)
        if with_interpolation_points:
            current_color = axis.lines[-1].get_color() # figure out current colour
            if dim == 1:
                axis.plot(self.x, self.f, marker='.', linestyle='', color=current_color)
            elif dim == 2:
                axis.plot(self.f[0], self.f[1], marker='.', linestyle='', color=current_color)
                axis.axis('equal')
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

    def compare(self, f, *args, **kwds):
        """
        Plots the original function against its chebfun interpolant.
        
        INPUTS:

            -- f: Python, Numpy, or Sage function
        """
        x   = np.linspace(-1, 1, 10000)
        fig = plt.figure()
        ax  = fig.add_subplot(211)
        
        ax.plot(x, f(x), '#dddddd', linewidth=10, label='Actual', *args, **kwds)
        label = 'Chebfun Interpolant (d={0})'.format(len(self))
        self.plot(color='red', label=label, *args, **kwds)
        ax.legend(loc='best')

        ax  = fig.add_subplot(212)
        ax.plot(x, abs(f(x)-self(x)), 'k')

        return ax

def chebpoly(n):
    if n == 0:
        return Chebfun(np.array([1.]))
    vals = np.ones(n+1)
    vals[-1::-2] = -1
    return Chebfun(vals)

def even_data(data):
    """
    Construct Extended Data Vector (equivalent to creating an
    even extension of the original function)
    """
    return np.vstack([data, data[-2:0:-1]])

def interpolation_points(N):
    """
    N+1 Chebyshev points in [-1, 1], boundaries included
    """
    if N == 0:
        return np.array([0.])
    return np.cos(np.arange(N+1)*np.pi/N)

def sample_function(f, N):
    """
    Sample a function on N+1 Chebyshev points.
    """
    x = interpolation_points(N)
    return f(x).T

def chebpolyfit(sampled):
    """
    Compute Chebyshev coefficients for values on N Chebyshev points.
    """
    if len(np.shape(sampled)) == 1: # make it a matrix
        sampled = np.reshape(sampled, (-1, 1))
    if len(sampled) == 1:
        return sampled[0]*np.array([1.]).reshape(-1, 1)
    evened = even_data(sampled)
    coeffs = dct(evened)
    return coeffs

import scipy.fftpack as fftpack

def dct(data):
    """
    Compute DCT
    """
    N = len(data)//2
    dctdata     = fftpack.dct(data[:N+1].T, 1).T
    dctdata     /= N
    dctdata[0]  /= 2.
    dctdata[-1] /= 2.
    return dctdata

def idct(chebcoeff):
    """
    Compute the inverse DCT
    """
    N = len(chebcoeff)
    if N == 1:
        return chebcoeff

    data = 2.*chebcoeff
    data[0] *= 2
    data[-1] *= 2
    data *= N

    idctdata = fftpack.dct(data.T, 1).T/(4*N)
    return idctdata

def interpolate(x, values):
    """
    Returns a polynomial with vector coefficients which interpolates the values at the points x
    """
    return Bary(x, values)



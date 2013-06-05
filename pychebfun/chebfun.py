#!/usr/bin/env python
# coding: UTF-8
"""
Chebfun module
==============

.. moduleauthor :: Chris Swierczewski <cswiercz@gmail.com>
.. moduleauthor :: Olivier Verdier <olivier.verdier@gmail.com>
.. moduleauthor :: Gregory Potter <ghpotter@gmail.com>


"""
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
        + (lots of) bug fixing + pythonisation
       """
    m = len(A)
    SA = (A.T* 2*np.arange(m)).T
    DA = np.zeros_like(A)
    if m == 1: # constant
        return np.zeros_like(A[0:1])
    if m == 2: # linear
        return A[1:2,]
    DA[m-3:m-1,] = SA[m-2:m,]
    for j in range(m//2 - 1):
        k = m-3-2*j
        DA[k] = SA[k+1] + DA[k+2]
        DA[k-1] = SA[k] + DA[k+1]
    DA[0] = (SA[1] + DA[2])*0.5
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

    class NoConvergence(Exception):
        """
        Raised when dichotomy does not converge.
        """

    def init_from_data(self, data):
        """
        The data provided are the values at the Chebyshev points
        """
        vals = np.asarray(data, dtype=float)
        N = len(vals)-1
        self.x = interpolation_points(N)
        self.f = vals.copy()
        self.p  = interpolate(self.x, self.f)

    def init_from_chebfun(self, other):
        """
        Initialise from another instance of Chebfun
        """
        self.x = other.x
        self.f = other.f
        self.p = other.p

    def init_from_chebcoeff(self, chebcoeff):
        """
        Initialise from provided Chebyshev coefficients
        """
        coeffs = np.asarray(chebcoeff)
        N = len(coeffs)
        self.f = idct(coeffs)
        self.x = interpolation_points(N-1)
        self.p = interpolate(self.x, self.f)

    def init_from_function(self, f, N=None):
        """
        Initialise from a function to sample.
        N: optional parameter which indicates the range of the dichotomy
        """
        args = {'f': f}
        if N is not None: # N is provided
            nextpow2 = int(np.log2(N))+1
            args['kmin'] = nextpow2
            args['kmax'] = nextpow2+1
            args['raise_no_convergence'] = False
        else:
            args['raise_no_convergence'] = True

        # Find out the right number of coefficients to keep
        coeffs, Nmax = dichotomy(**args)


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

    def cos(self):
        return Chebfun(lambda x: np.cos(self(x)))

    def exp(self):
        return Chebfun(lambda x: np.exp(self(x)))


    #
    # Numpy / Scipy Operator Overloads
    #

    def chebyshev_coefficients(self):
        return chebpolyfit(self.f)

    def sum(self):
        """
        Evaluate the integral of the Chebfun over the given interval using
        Clenshaw-Curtis quadrature.
        """
        ai = self.chebyshev_coefficients()
        ai2 = ai[::2]
        n = len(ai2)
        Tints = 2/(1-(2*np.arange(n))**2)
        val = np.sum((Tints*ai2.T).T, axis=0)

        return val

    def norm(self):
        """
        Return: square root of integral of |f|**2 over [-1,1]
        """
        square = self*self
        integral = square.sum()
        norm = np.sqrt(integral)
        return norm

    def integrate(self):
        """
        Return the Chebfun representing the integral of self over the domain.

        (Simply numerically integrates the underlying Barcentric polynomial.)
        """
        return Chebfun(self.p.integrate)

 
    def derivative(self):
        return self.differentiate()

    def differentiate(self):
        bi = differentiator(self.chebyshev_coefficients())
        return Chebfun(chebcoeff=bi)

    def roots(self):
        """
        Return the roots if the Chebfun is scalar
        """
        ai = self.chebyshev_coefficients()
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

        coeffs = self.chebyshev_coefficients()
        data = np.log10(np.abs(coeffs))
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

def basis(n):
    if n == 0:
        return Chebfun(np.array([1.]))
    vals = np.ones(n+1)
    vals[-1::-2] = -1
    return Chebfun(vals)

def dichotomy(f, kmin=2, kmax=12, raise_no_convergence=True):
    """
    Compute the coefficients for a function f by dichotomy.
    kmin, kmax: log2 of number of interpolation points to try
    raise_no_convergence: whether to raise an exception if the dichotomy does not converge
    """

    for k in xrange(kmin, kmax):
        N = pow(2, k)

        sampled = sample_function(f, N)
        coeffs = chebpolyfit(sampled)

        # 3) Check for negligible coefficients
        #    If within bound: get negligible coeffs and bread
        bnd = 128*emach*abs(np.max(coeffs))

        last = abs(coeffs[-2:])
        if np.all(last <= bnd):
            break
    else:
        if raise_no_convergence:
            raise Chebfun.NoConvergence(last, bnd)
    inds  = np.nonzero(abs(coeffs) >= bnd)
    Nmax = inds[0][-1]
    return coeffs, Nmax

def even_data(data):
    """
    Construct Extended Data Vector (equivalent to creating an
    even extension of the original function)
    """
    return np.concatenate([data, data[-2:0:-1]],)

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
    Compute Chebyshev coefficients for values located on Chebyshev points.
    sampled: array; first dimension is number of Chebyshev points
    """
    asampled = np.asarray(sampled)
    if len(asampled) == 1:
        return asampled
    evened = even_data(asampled)
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



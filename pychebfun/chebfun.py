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
import numpy.polynomial as poly

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

def chebfun(f=None, N=None, chebcoeff=None,):
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
        return Chebfun(f)

    if isinstance(f, Chebfun): # copy if f is another Chebfun
        return Chebfun.from_chebfun(f)

    if chebcoeff is not None: # if the coefficients of a Chebfun are given
        return Chebfun.from_chebcoeff(chebcoeff)

    # from this point, we assume that f is a function
    if f is not None:
        return Chebfun.from_function(f, N)

    return Chebfun([])




class Chebfun(object):
    """
    Construct a Lagrange interpolating polynomial over the Chebyshev points.

    """

    class NoConvergence(Exception):
        """
        Raised when dichotomy does not converge.
        """

    @classmethod
    def from_data(self, data):
        return self(data)

    @classmethod
    def from_chebfun(self, other):
        """
        Initialise from another instance of Chebfun
        """
        return self(other.values())

    @classmethod
    def from_chebcoeff(self, chebcoeff, prune=True, scale=1.):
        """
        Initialise from provided Chebyshev coefficients
        prune: Whether to prune the negligible coefficients
        scale: the scale to use when pruning
        """
        coeffs = np.asarray(chebcoeff)
        if prune:
            N = self._cutoff(coeffs, scale)
            pruned_coeffs = coeffs[:N]
        else:
            pruned_coeffs = coeffs
        values = chebpolyval(pruned_coeffs)
        return self(values, scale)

    @classmethod
    def dichotomy(self, f, kmin=2, kmax=12, raise_no_convergence=True,):
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
            bnd = self._threshold(abs(np.max(coeffs)))

            last = abs(coeffs[-2:])
            if np.all(last <= bnd):
                break
        else:
            if raise_no_convergence:
                raise self.NoConvergence(last, bnd)
        return coeffs

    @classmethod
    def from_function(self, f, N=None):
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
        coeffs = self.dichotomy(**args)

        return self.from_chebcoeff(coeffs,)

    @classmethod
    def _threshold(self, scale):
        bnd = 128*emach*scale
        return bnd

    @classmethod
    def _cutoff(self, coeffs, scale):
        """
        Compute cutoff index after which the coefficients are deemed negligible.
        """
        bnd = self._threshold(scale)
        inds  = np.nonzero(abs(coeffs) >= bnd)
        if len(inds[0]):
            N = inds[0][-1]
        else:
            N = 0
        return N+1

    def __init__(self, values=0., scale=None):
        """
        Init a Chebfun objects from values at Chebyshev points.
        values: Interpolation values
        scale: The actual scale; computed automatically if not given
        """
        avalues = np.asarray(values, dtype=float)
        avalues1 = np.atleast_1d(avalues)
        N = len(avalues1)
        points = interpolation_points(N)
        self._values = avalues1
        if scale is not None:
            self._scale = scale
        else:
            self._scale = np.max(np.abs(self._values))
        self.p = interpolate(points, avalues1)

    def __repr__(self):
        return "<Chebfun({0})>".format(repr(self.values()))

    def __str__(self):
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
        ps = [self, other]
        # length difference
        diff = len(other) - len(self)
        # determine which of self/other is the smaller/bigger
        big = diff > 0
        small = not big
        # pad the chebyshev coefficients of the small one with zeros
        small_coeffs = ps[small].chebyshev_coefficients()
        big_coeffs = ps[big].chebyshev_coefficients()
        padded = np.zeros_like(big_coeffs)
        padded[:len(small_coeffs)] = small_coeffs
        # add the values and create a new Chebfun with them
        chebsum = big_coeffs + padded
        new_scale = np.max([self._scale, other._scale])
        return self.from_chebcoeff(chebsum, scale=new_scale)

    __radd__ = __add__


    @cast_scalar
    def __sub__(self, other):
        """
        Chebfun subtraction.
        """
        return self + (-other)

    def __rsub__(self, other):
        return -(self - other)


    @cast_scalar
    def __mul__(self, other):
        """
        Chebfun multiplication.
        """
        return self.from_function(lambda x: self(x) * other(x),)

    __rmul__ = __mul__

    @cast_scalar
    def __div__(self, other):
        """
        Chebfun division
        """
        return self.from_function(lambda x: self(x) / other(x),)

    __truediv__ = __div__

    @cast_scalar
    def __rdiv__(self, other):
        return self.from_function(lambda x: other(x)/self(x))

    __rtruediv__ = __rdiv__

    def __neg__(self):
        """
        Chebfun negation.
        """
        return self.from_data(-self.values())

    def __pow__(self, other):
        return self.from_function(lambda x: self(x)**other)

    def __abs__(self):
        return self.from_function(lambda x: abs(self(x)))

    #
    # Numpy / Scipy Operator Overloads
    #

    def chebyshev_coefficients(self):
        return chebpolyfit(self.values())

    def values(self):
        return self._values

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

    def dot(self, other):
        """
        Return the Hilbert scalar product $\int f.g$.
        """
        prod = self * other
        return prod.sum()

    def norm(self):
        """
        Return: square root of scalar product with itself.
        """
        norm = np.sqrt(self.dot(self))
        return norm

    def integrate(self):
        """
        Return the Chebfun representing the primitive of self over the domain, starting at zero.
        """
        coeffs = self.chebyshev_coefficients()
        int_coeffs = poly.chebyshev.chebint(coeffs)
        return self.from_chebcoeff(int_coeffs)

    def derivative(self):
        return self.differentiate()

    def differentiate(self, n=1):
        """
        n-th derivative
        """
        bi = self.chebyshev_coefficients()
        for _ in range(n):
            bi = differentiator(bi)
        return self.from_chebcoeff(chebcoeff=bi)

    def roots(self):
        """
        Return the roots if the Chebfun is scalar
        The computation is done via trigonometric polynomials
        """
        ai = self.chebyshev_coefficients()
        N = len(ai)
        coeffs = np.hstack([ai[-1::-1], ai[1:]])
        coeffs[N-1] *= 2
        complex_roots = poly.polynomial.polyroots(coeffs)
        real_roots = np.array([np.real(r) for r in complex_roots if np.allclose(abs(r), 1.)])
        roots = np.unique(real_roots)
        return roots

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
                axis.plot(self.p.xi, self.values(), marker='.', linestyle='', color=current_color)
            elif dim == 2:
                axis.plot(self.values()[:, 0], self.values()[:, 1], marker='.', linestyle='', color=current_color)
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
        plt.plot(self.p.xi, self.values())

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

def _add_delegate(ufunc):
    def method(self):
        return self.from_function(lambda x: ufunc(self(x)))
    name = ufunc.__name__
    method.__name__ = name
    method.__doc__ = "delegate for numpy's ufunc {}".format(name)
    setattr(Chebfun, name, method)

# Following list generated from:
# https://github.com/qsnake/numpy/blob/master/numpy/core/code_generators/generate_umath.py
for func in [np.arccos, np.arccosh, np.arcsin, np.arcsinh, np.arctan, np.arctanh, np.cos, np.sin, np.tan, np.cosh, np.sinh, np.tanh, np.exp, np.exp2, np.expm1, np.log, np.log2, np.log1p, np.sqrt, np.ceil, np.trunc, np.fabs, np.floor, ]:
    _add_delegate(func)


def basis(n):
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
    return np.concatenate([data, data[-2:0:-1]],)

def interpolation_points(N):
    """
    N Chebyshev points in [-1, 1], boundaries included
    """
    if N == 1:
        return np.array([0.])
    return np.cos(np.arange(N)*np.pi/(N-1))

def sample_function(f, N):
    """
    Sample a function on N+1 Chebyshev points.
    """
    x = interpolation_points(N+1)
    return f(x)

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
    Compute DCT using FFT
    """
    N = len(data)//2
    fftdata     = fftpack.fft(data, axis=0)[:N+1]
    fftdata     /= N
    fftdata[0]  /= 2.
    fftdata[-1] /= 2.
    if np.isrealobj(data):
        data = np.real(fftdata)
    else:
        data = fftdata
    return data

def chebpolyval(chebcoeff):
    """
    Compute the interpolation values at Chebyshev points.
    chebcoeff: Chebyshev coefficients
    """
    N = len(chebcoeff)
    if N == 1:
        return chebcoeff

    data = even_data(chebcoeff)/2
    data[0] *= 2
    data[N-1] *= 2

    fftdata = 2*(N-1)*fftpack.ifft(data, axis=0)
    complex_values = fftdata[:N]
    # convert to real if input was real
    if np.isrealobj(chebcoeff):
        values = np.real(complex_values)
    else:
        values = complex_values
    return values

def interpolate(x, values):
    """
    Returns a polynomial with vector coefficients which interpolates the values at the points x
    """
    return Bary(x, values)


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


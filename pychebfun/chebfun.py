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


import operator

import numpy as np
from scipy import linalg

from scipy.interpolate import BarycentricInterpolator as Bary
import numpy.polynomial as poly
import scipy.fftpack as fftpack

from .polyfun import Polyfun, cast_scalar

class Chebfun(Polyfun):
    """
    Eventually set this up so that a Chebfun is a collection of Chebfuns. This
    will enable piecewise smooth representations al la Matlab Chebfun v2.0.
    """
    # ----------------------------------------------------------------
    # Standard construction class methods.
    # ----------------------------------------------------------------

    @classmethod
    def get_default_domain(self, domain=None):
        if domain is None:
            return [-1., 1.]
        else:
            return domain

    @classmethod
    def identity(self, domain=[-1., 1.]):
        """
        The identity function x -> x.
        """
        return self.from_data([domain[1],domain[0]], domain)

    @classmethod
    def basis(self, n):
        """
        Chebyshev basis functions T_n.
        """
        if n == 0:
            return self(np.array([1.]))
        vals = np.ones(n+1)
        vals[1::2] = -1
        return self(vals)

    # ----------------------------------------------------------------
    # Integration and differentiation
    # ----------------------------------------------------------------

    def sum(self):
        """
        Evaluate the integral over the given interval using
        Clenshaw-Curtis quadrature.
        """
        ak = self.coefficients()
        ak2 = ak[::2]
        n = len(ak2)
        Tints = 2/(1-(2*np.arange(n))**2)
        val = np.sum((Tints*ak2.T).T, axis=0)
        a_, b_ = self.domain()
        return 0.5*(b_-a_)*val

    def integrate(self):
        """
        Return the object representing the primitive of self over the domain. The
        output starts at zero on the left-hand side of the domain.
        """
        coeffs = self.coefficients()
        a,b = self.domain()
        int_coeffs = 0.5*(b-a)*poly.chebyshev.chebint(coeffs)
        antiderivative = self.from_coeff(int_coeffs, domain=self.domain())
        return antiderivative - antiderivative(a)

    def differentiate(self, n=1):
        """
        n-th derivative, default 1.
        """
        ak = self.coefficients()
        a_, b_ = self.domain()
        for _ in range(n):
            ak = self.differentiator(ak)
        return self.from_coeff((2./(b_-a_))**n*ak, domain=self.domain())

    # ----------------------------------------------------------------
    # Roots
    # ----------------------------------------------------------------
    def roots(self):
        """
        Utilises Boyd's O(n^2) recursive subdivision algorithm. The chebfun
        is recursively subsampled until it is successfully represented to
        machine precision by a sequence of piecewise interpolants of degree
        100 or less. A colleague matrix eigenvalue solve is then applied to
        each of these pieces and the results are concatenated.

        See:
        J. P. Boyd, Computing zeros on a real interval through Chebyshev
        expansion and polynomial rootfinding, SIAM J. Numer. Anal., 40
        (2002), pp. 1666–1682.
        """
        if self.size() == 1:
            return np.array([])

        elif self.size() <= 100:
            ak = self.coefficients()
            v = np.zeros_like(ak[:-1])
            v[1] = 0.5
            C1 = linalg.toeplitz(v)
            C2 = np.zeros_like(C1)
            C1[0,1] = 1.
            C2[-1,:] = ak[:-1]
            C = C1 - .5/ak[-1] * C2
            eigenvalues = linalg.eigvals(C)
            roots = [eig.real for eig in eigenvalues
                    if np.allclose(eig.imag,0,atol=1e-10)
                        and np.abs(eig.real) <=1]
            scaled_roots = self._ui_to_ab(np.array(roots))
            return scaled_roots
        else:
            # divide at a close-to-zero split-point
            split_point = self._ui_to_ab(0.0123456789)
            return np.concatenate(
                (self.restrict([self._domain[0],split_point]).roots(),
                 self.restrict([split_point,self._domain[1]]).roots())
            )

    # ----------------------------------------------------------------
    # Interpolation and evaluation (go from values to coefficients)
    # ----------------------------------------------------------------

    @classmethod
    def interpolation_points(self, N):
        """
        N Chebyshev points in [-1, 1], boundaries included
        """
        if N == 1:
            return np.array([0.])
        return np.cos(np.arange(N)*np.pi/(N-1))

    @classmethod
    def sample_function(self, f, N):
        """
        Sample a function on N+1 Chebyshev points.
        """
        x = self.interpolation_points(N+1)
        return f(x)

    @classmethod
    def polyfit(self, sampled):
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

    @classmethod
    def polyval(self, chebcoeff):
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

    @classmethod
    def interpolator(self, x, values):
        """
        Returns a polynomial with vector coefficients which interpolates the values at the Chebyshev points x
        """
        # hacking the barycentric interpolator by computing the weights in advance
        p = Bary([0.])
        N = len(values)
        weights = np.ones(N)
        weights[0] = .5
        weights[1::2] = -1
        weights[-1] *= .5
        p.wi = weights
        p.xi = x
        p.set_yi(values)
        return p

    # ----------------------------------------------------------------
    # Helper for differentiation.
    # ----------------------------------------------------------------

    @classmethod
    def differentiator(self, A):
        """Differentiate a set of Chebyshev polynomial expansion
           coefficients
           Originally from http://www.scientificpython.net/pyblog/chebyshev-differentiation
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

# ----------------------------------------------------------------
# General utilities
# ----------------------------------------------------------------

def even_data(data):
    """
    Construct Extended Data Vector (equivalent to creating an
    even extension of the original function)
    Return: array of length 2(N-1)
    For instance, [0,1,2,3,4] --> [0,1,2,3,4,3,2,1]
    """
    return np.concatenate([data, data[-2:0:-1]],)

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

# ----------------------------------------------------------------
# Add overloaded operators
# ----------------------------------------------------------------

def _add_operator(cls, op):
    def method(self, other):
        if not self.same_domain(other):
            raise self.DomainMismatch(self.domain(), other.domain())
        return self.from_function(
            lambda x: op(self(x).T, other(x).T).T, domain=self.domain(), )
    cast_method = cast_scalar(method)
    name = '__'+op.__name__+'__'
    cast_method.__name__ = name
    cast_method.__doc__ = "operator {}".format(name)
    setattr(cls, name, cast_method)

def rdiv(a, b):
    return b/a

for _op in [operator.mul, operator.truediv, operator.pow, rdiv]:
    _add_operator(Polyfun, _op)

# ----------------------------------------------------------------
# Add numpy ufunc delegates
# ----------------------------------------------------------------

def _add_delegate(ufunc, nonlinear=True):
    def method(self):
        return self.from_function(lambda x: ufunc(self(x)), domain=self.domain())
    name = ufunc.__name__
    method.__name__ = name
    method.__doc__ = "delegate for numpy's ufunc {}".format(name)
    setattr(Polyfun, name, method)

# Following list generated from:
# https://github.com/numpy/numpy/blob/master/numpy/core/code_generators/generate_umath.py
for func in [np.arccos, np.arccosh, np.arcsin, np.arcsinh, np.arctan, np.arctanh, np.cos, np.sin, np.tan, np.cosh, np.sinh, np.tanh, np.exp, np.exp2, np.expm1, np.log, np.log2, np.log1p, np.sqrt, np.ceil, np.trunc, np.fabs, np.floor, ]:
    _add_delegate(func)


# ----------------------------------------------------------------
# General Aliases
# ----------------------------------------------------------------
## chebpts = interpolation_points

# ----------------------------------------------------------------
# Constructor inspired by the Matlab version
# ----------------------------------------------------------------

def chebfun(f=None, domain=[-1,1], N=None, chebcoeff=None,):
    """
    Create a Chebyshev polynomial approximation of the function $f$ on the interval :math:`[-1, 1]`.

    :param callable f: Python, Numpy, or Sage function
    :param int N: (default = None)  specify number of interpolating points
    :param np.array chebcoeff: (default = np.array(0)) specify the coefficients
    """

    # Chebyshev coefficients
    if chebcoeff is not None:
        return Chebfun.from_coeff(chebcoeff, domain)

    # another instance
    if isinstance(f, Polyfun):
        return Chebfun.from_fun(f)

    # callable
    if hasattr(f, '__call__'):
        return Chebfun.from_function(f, domain, N)

    # from here on, assume that f is None, or iterable
    if np.isscalar(f):
        f = [f]

    try:
        iter(f) # interpolation values provided
    except TypeError:
        pass
    else:
        return Chebfun(f, domain)

    raise TypeError('Impossible to initialise the object from an object of type {}'.format(type(f)))

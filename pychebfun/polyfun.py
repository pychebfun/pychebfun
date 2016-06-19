#!/usr/bin/env python
# coding: UTF-8
from __future__ import division


import numpy as np

import sys
emach = sys.float_info.epsilon                        # machine epsilon

from functools import wraps

def cast_scalar(method):
    """
    Cast scalars to constant interpolating objects
    """
    @wraps(method)
    def new_method(self, other):
        if np.isscalar(other):
            other = type(self)([other],self.domain())
        return method(self, other)
    return new_method




class Polyfun(object):
    """
    Construct a Lagrange interpolating polynomial over arbitrary points.
    Polyfun objects consist in essence of two components:

        1) An interpolant on [-1,1],
        2) A domain attribute [a,b].

    These two pieces of information are used to define and subsequently
    keep track of operations upon Chebyshev interpolants defined on an
    arbitrary real interval [a,b].

    """

    # ----------------------------------------------------------------
    # Initialisation methods
    # ----------------------------------------------------------------

    class NoConvergence(Exception):
        """
        Raised when dichotomy does not converge.
        """

    class DomainMismatch(Exception):
        """
        Raised when there is an interval mismatch.
        """

    @classmethod
    def from_data(self, data, domain=None):
        """
        Initialise from interpolation values.
        """
        return self(data,domain)

    @classmethod
    def from_fun(self, other):
        """
        Initialise from another instance
        """
        return self(other.values(),other.domain())

    @classmethod
    def from_coeff(self, chebcoeff, domain=None, prune=True, vscale=1.):
        """
        Initialise from provided coefficients
        prune: Whether to prune the negligible coefficients
        vscale: the scale to use when pruning
        """
        coeffs = np.asarray(chebcoeff)
        if prune:
            N = self._cutoff(coeffs, vscale)
            pruned_coeffs = coeffs[:N]
        else:
            pruned_coeffs = coeffs
        values = self.polyval(pruned_coeffs)
        return self(values, domain, vscale)

    @classmethod
    def dichotomy(self, f, kmin=2, kmax=12, raise_no_convergence=True,):
        """
        Compute the coefficients for a function f by dichotomy.
        kmin, kmax: log2 of number of interpolation points to try
        raise_no_convergence: whether to raise an exception if the dichotomy does not converge
        """

        for k in range(kmin, kmax):
            N = pow(2, k)

            sampled = self.sample_function(f, N)
            coeffs = self.polyfit(sampled)

            # 3) Check for negligible coefficients
            #    If within bound: get negligible coeffs and bread
            bnd = self._threshold(np.max(np.abs(coeffs)))

            last = abs(coeffs[-2:])
            if np.all(last <= bnd):
                break
        else:
            if raise_no_convergence:
                raise self.NoConvergence(last, bnd)
        return coeffs

    @classmethod
    def from_function(self, f, domain=None, N=None):
        """
        Initialise from a function to sample.
        N: optional parameter which indicates the range of the dichotomy
        """
        # rescale f to the unit domain
        domain = self.get_default_domain(domain)
        a,b = domain[0], domain[-1]
        map_ui_ab = lambda t: 0.5*(b-a)*t + 0.5*(a+b)
        args = {'f': lambda t: f(map_ui_ab(t))}
        if N is not None: # N is provided
            nextpow2 = int(np.log2(N))+1
            args['kmin'] = nextpow2
            args['kmax'] = nextpow2+1
            args['raise_no_convergence'] = False
        else:
            args['raise_no_convergence'] = True

        # Find out the right number of coefficients to keep
        coeffs = self.dichotomy(**args)

        return self.from_coeff(coeffs, domain)

    @classmethod
    def _threshold(self, vscale):
        """
        Compute the threshold at which coefficients are trimmed.
        """
        bnd = 128*emach*vscale
        return bnd

    @classmethod
    def _cutoff(self, coeffs, vscale):
        """
        Compute cutoff index after which the coefficients are deemed negligible.
        """
        bnd = self._threshold(vscale)
        inds  = np.nonzero(abs(coeffs) >= bnd)
        if len(inds[0]):
            N = inds[0][-1]
        else:
            N = 0
        return N+1


    def __init__(self, values=0., domain=None, vscale=None):
        """
        Init an object from values at interpolation points.
        values: Interpolation values
        vscale: The actual vscale; computed automatically if not given
        """
        avalues = np.asarray(values,)
        avalues1 = np.atleast_1d(avalues)
        N = len(avalues1)
        points = self.interpolation_points(N)
        self._values = avalues1
        if vscale is not None:
            self._vscale = vscale
        else:
            self._vscale = np.max(np.abs(self._values))
        self.p = self.interpolator(points, avalues1)

        domain = self.get_default_domain(domain)
        self._domain = np.array(domain)
        a,b = domain[0], domain[-1]

        # maps from [-1,1] <-> [a,b]
        self._ab_to_ui = lambda x: (2.0*x-a-b)/(b-a)
        self._ui_to_ab = lambda t: 0.5*(b-a)*t + 0.5*(a+b)

    def same_domain(self, fun2):
        """
        Returns True if the domains of two objects are the same.
        """
        return np.allclose(self.domain(), fun2.domain(), rtol=1e-14, atol=1e-14)

    # ----------------------------------------------------------------
    # String representations
    # ----------------------------------------------------------------

    def __repr__(self):
        """
        Display method
        """
        a, b = self.domain()
        vals = self.values()
        return (
            '%s \n '
            '    domain        length     endpoint values\n '
            ' [%5.1f, %5.1f]     %5d       %5.2f   %5.2f\n '
            'vscale = %1.2e') % (
                str(type(self)).split('.')[-1].split('>')[0][:-1],
                a,b,self.size(),vals[-1],vals[0],self._vscale,)

    def __str__(self):
        return "<{0}({1})>".format(
            str(type(self)).split('.')[-1].split('>')[0][:-1],self.size(),)

    # ----------------------------------------------------------------
    # Basic Operator Overloads
    # ----------------------------------------------------------------

    def __call__(self, x):
        return self.p(self._ab_to_ui(x))

    def __getitem__(self, s):
        """
        Components s of the fun.
        """
        return self.from_data(self.values().T[s].T)

    def __bool__(self):
        """
        Test for difference from zero (up to tolerance)
        """
        return not np.allclose(self.values(), 0)

    __nonzero__ = __bool__

    def __eq__(self, other):
        return not(self - other)

    def __ne__(self, other):
        return not (self == other)

    @cast_scalar
    def __add__(self, other):
        """
        Addition
        """
        if not self.same_domain(other):
            raise self.DomainMismatch(self.domain(),other.domain())

        ps = [self, other]
        # length difference
        diff = other.size() - self.size()
        # determine which of self/other is the smaller/bigger
        big = diff > 0
        small = not big
        # pad the coefficients of the small one with zeros
        small_coeffs = ps[small].coefficients()
        big_coeffs = ps[big].coefficients()
        padded = np.zeros_like(big_coeffs)
        padded[:len(small_coeffs)] = small_coeffs
        # add the values and create a new object with them
        chebsum = big_coeffs + padded
        new_vscale = np.max([self._vscale, other._vscale])
        return self.from_coeff(
            chebsum, domain=self.domain(), vscale=new_vscale
        )

    __radd__ = __add__


    @cast_scalar
    def __sub__(self, other):
        """
        Subtraction.
        """
        return self + (-other)

    def __rsub__(self, other):
        return -(self - other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __neg__(self):
        """
        Negation.
        """
        return self.from_data(-self.values(),domain=self.domain())


    def __abs__(self):
        return self.from_function(lambda x: abs(self(x)),domain=self.domain())

    # ----------------------------------------------------------------
    # Attributes
    # ----------------------------------------------------------------

    def size(self):
        return self.p.n

    def coefficients(self):
        return self.polyfit(self.values())

    def values(self):
        return self._values

    def domain(self):
        return self._domain

    # ----------------------------------------------------------------
    # Integration and differentiation
    # ----------------------------------------------------------------

    def integrate(self):
        raise NotImplementedError()

    def differentiate(self):
        raise NotImplementedError()

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


    # ----------------------------------------------------------------
    # Miscellaneous operations
    # ----------------------------------------------------------------
    def restrict(self,subinterval):
        """
        Return a Polyfun that matches self on subinterval.
        """
        if (subinterval[0] < self._domain[0]) or (subinterval[1] > self._domain[1]):
            raise ValueError("Can only restrict to subinterval")
        return self.from_function(self, subinterval)


    # ----------------------------------------------------------------
    # Class method aliases
    # ----------------------------------------------------------------
    diff = differentiate
    cumsum = integrate

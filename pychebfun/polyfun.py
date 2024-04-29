#!/usr/bin/env python

from typing import Self, Optional, Callable

import scipy.interpolate

import numpy as np
import numpy.typing as npt

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



class Polyfun:
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
    def from_data(cls, data: npt.ArrayLike, domain:Optional[tuple]=None) -> Self:
        """
        Initialise from interpolation values.
        """
        return cls(data,domain)

    @classmethod
    def from_fun(cls, other: Self) -> Self:
        """
        Initialise from another instance
        """
        return cls(other.values(),other.domain())

    @classmethod
    def from_coeff(cls, chebcoeff: npt.ArrayLike, domain:Optional[tuple]=None, prune: bool=True, vscale: float=1.):
        """
        Initialise from provided coefficients
        prune: Whether to prune the negligible coefficients
        vscale: the scale to use when pruning
        """
        coeffs = np.asarray(chebcoeff)
        if prune:
            N = cls._cutoff(coeffs, vscale)
            pruned_coeffs = coeffs[:N]
        else:
            pruned_coeffs = coeffs
        values = cls.polyval(pruned_coeffs)
        return cls(values, domain, vscale)

    @classmethod
    def dichotomy(cls, f: Callable, kmin: int=2, kmax :int=12, raise_no_convergence:bool=True,) -> np.ndarray:
        """
        Compute the coefficients for a function f by dichotomy.
        kmin, kmax: log2 of number of interpolation points to try
        raise_no_convergence: whether to raise an exception if the dichotomy does not converge
        """
        last = [0,0]
        bnd = 0
        coeffs = np.zeros(1)
        for k in range(kmin, kmax):
            N = pow(2, k)

            sampled = cls.sample_function(f, N)
            coeffs = cls.polyfit(sampled)

            # 3) Check for negligible coefficients
            #    If within bound: get negligible coeffs and bread
            bnd = cls._threshold(np.max(np.abs(coeffs)))

            last = abs(coeffs[-2:])
            if np.all(last <= bnd):
                break
        else:
            if raise_no_convergence:
                raise cls.NoConvergence(last, bnd)
        return coeffs

    @classmethod
    def get_default_domain(cls, domain:Optional[tuple]=None) -> tuple:
        if domain is None:
            return (-1., 1.)
        else:
            return domain


    @classmethod
    def from_function(cls,
                      f: Callable[[npt.ArrayLike], npt.ArrayLike],
                      domain:Optional[tuple]=None,
                      N:Optional[int]=None) -> Self:
        """
        Initialise from a function to sample.
        N: optional parameter which indicates the range of the dichotomy
        """
        # rescale f to the unit domain
        a,b = cls.get_default_domain(domain)
        map_ui_ab = lambda t: 0.5*(b-a)*t + 0.5*(a+b)
        args = {}
        args['f'] = lambda t: f(map_ui_ab(t))
        if N is not None: # N is provided
            nextpow2 = int(np.log2(N))+1
            args['kmin'] = nextpow2
            args['kmax'] = nextpow2+1
            args['raise_no_convergence'] = False
        else:
            args['raise_no_convergence'] = True

        # Find out the right number of coefficients to keep
        coeffs = cls.dichotomy(**args)

        return cls.from_coeff(coeffs, domain)

    @classmethod
    def _threshold(cls, vscale: float) -> float:
        """
        Compute the threshold at which coefficients are trimmed.
        """
        bnd = 128*emach*vscale
        return bnd

    @classmethod
    def _cutoff(cls, coeffs:np.ndarray, vscale:float) -> int:
        """
        Compute cutoff index after which the coefficients are deemed negligible.
        """
        bnd = cls._threshold(vscale)
        inds  = np.nonzero(abs(coeffs) >= bnd)
        if len(inds[0]):
            N = inds[0][-1]
        else:
            N = 0
        return N+1


    def __init__(self, values:npt.ArrayLike=0., domain:Optional[tuple]=None, vscale:Optional[float]=None):
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
        self._domain = domain
        a,b = domain[0], domain[-1]

        # maps from [-1,1] <-> [a,b]
        self._ab_to_ui = lambda x: (2.0*x-a-b)/(b-a)
        self._ui_to_ab = lambda t: 0.5*(b-a)*t + 0.5*(a+b)

    def same_domain(self, fun2: Self) -> bool:
        """
        Returns True if the domains of two objects are the same.
        """
        return np.allclose(self.domain(), fun2.domain(), rtol=1e-14, atol=1e-14)

    # ----------------------------------------------------------------
    # String representations
    # ----------------------------------------------------------------

    def __repr__(self) -> str:
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

    def __str__(self) -> str:
        return "<{0}({1})>".format(
            str(type(self)).split('.')[-1].split('>')[0][:-1],self.size(),)

    # ----------------------------------------------------------------
    # Basic Operator Overloads
    # ----------------------------------------------------------------

    def __call__(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return self.p(self._ab_to_ui(x))

    def __getitem__(self, s:int) -> Self:
        """
        Components s of the fun.
        """
        return self.from_data(self.values().T[s].T)

    def __bool__(self) -> bool:
        """
        Test for difference from zero (up to tolerance)
        """
        return not np.allclose(self.values(), 0)

    __nonzero__ = __bool__

    def __eq__(self, other:Self) -> bool:
        return not(self - other)

    def __ne__(self, other:Self) -> bool:
        return not (self == other)

    @cast_scalar
    def __add__(self, other: Self) -> Self:
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


    def shift(self, x: complex) -> Self:
        return type(self).from_data(self.values() + x)


    @cast_scalar
    def __sub__(self, other: Self) -> Self:
        """
        Subtraction.
        """
        return self + (-other)

    def __rsub__(self, other: Self) -> Self:
        return -(self - other)

    def __rmul__(self, other: Self) -> Self:
        return self.__mul__(other)

    def __rtruediv__(self, other: Self) -> Self:
        return self.__rdiv__(other)

    def __neg__(self) -> Self:
        """
        Negation.
        """
        return self.from_data(-self.values(),domain=self.domain())


    def __abs__(self):
        def abs_self(x: npt.ArrayLike) -> npt.ArrayLike:
            return np.abs(self(x))
        return self.from_function(abs_self, domain=self.domain())

    # ----------------------------------------------------------------
    # Attributes
    # ----------------------------------------------------------------

    def size(self) -> int:
        return self.p.n

    def coefficients(self) -> np.ndarray:
        return self.polyfit(self.values())

    def values(self) -> np.ndarray:
        return self._values

    def domain(self) -> tuple:
        return self._domain

    # ----------------------------------------------------------------
    # Integration and differentiation
    # ----------------------------------------------------------------

    def integrate(self):
        raise NotImplementedError()

    def differentiate(self):
        raise NotImplementedError()

    def dot(self, other: Self) -> complex:
        """
        Return the Hilbert scalar product $∫f.g$.
        """
        prod = self * other
        return prod.sum()

    def sum(self) -> complex:
        raise NotImplementedError()

    def norm(self: Self) -> float:
        """
        Return: square root of scalar product with itself.
        """
        norm = np.sqrt(self.dot(self))
        return norm


    # ----------------------------------------------------------------
    # Miscellaneous operations
    # ----------------------------------------------------------------
    def restrict(self, subinterval:tuple) -> Self:
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

    @classmethod
    def polyval(cls, chebcoeffs:np.ndarray) -> np.ndarray:
        raise NotImplementedError()


    @classmethod
    def sample_function(cls, f: Callable, N:int) -> np.ndarray:
        raise NotImplementedError()

    @classmethod
    def polyfit(cls, sampled: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @classmethod
    def interpolation_points(cls, N:int) -> np.ndarray:
        raise NotImplementedError()

    @classmethod
    def interpolator(cls, x, values: np.ndarray) -> scipy.interpolate.BarycentricInterpolator:
        raise NotImplementedError()

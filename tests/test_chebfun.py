#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import os

import sys
testdir = os.path.dirname(__file__)
moduledir = os.path.join(testdir, os.path.pardir)
sys.path.insert(0, moduledir)
from pychebfun import *

import numpy as np
np.seterr(all='raise')
import numpy.testing as npt

import unittest

def Identity(x):
    return x

def One(x):
    return np.ones_like(x, dtype=float)

def Zero(x):
    return np.zeros_like(x, dtype=float)

from tools import *

def segment(x):
    y = np.expand_dims(x, axis=-1)
    zeros = np.zeros_like(y)
    return np.concatenate([y, zeros], axis=-1)

class TestSegment(unittest.TestCase):
    def setUp(self):
        self.fun = segment

    def test_shape(self):
        val = self.fun(0.)
        self.assertEqual(val.shape, (2,))
        valv = self.fun(np.arange(3.))
        self.assertEqual(valv.shape, (3,2))

class TestCircle(TestSegment):
    def setUp(self):
        self.fun = circle

def Quad(x):
    return x*x

def piecewise_continuous(x):
    """
    The function is on the verge of being discontinuous at many points
    """
    return np.exp(x)*np.sin(3*x)*np.tanh(5*np.cos(30*x))

def runge(x):
    return 1./(1+25*x**2)

xs = np.linspace(-1, 1, 1000)

class Test_Chebfun(unittest.TestCase):
    def setUp(self):
        # Constuct the O(dx^-16) "spectrally accurate" chebfun p
        self.p = Chebfun.from_function(f)

    def test_biglen(self):
        self.assertGreaterEqual(len(self.p), 4)

    def test_len(self):
        """
        Length of chebfun is equal to the number of Cheb coefficients (i.e., degree)
        """
        self.assertEqual(len(self.p), len(self.p.chebyshev_coefficients()))

    def test_error(self):
        """
        Chebfun is closed to function f up to tolerance
        """
        x = xs
        err = abs(f(x)-self.p(x))
        npt.assert_array_almost_equal(self.p(x),f(x),decimal=13)

    def test_root(self):
        """
        Roots are zeros of the chebfun.
        """
        roots = self.p.roots()
        npt.assert_array_almost_equal(f(roots),0)

    def test_all_roots(self):
        """
        Capture all rots.
        """
        roots = self.p.roots()
        self.assertEqual(len(roots),22)

    def test_chebcoeff(self):
        new = Chebfun.from_chebcoeff(self.p.chebyshev_coefficients())
        npt.assert_allclose(self.p(xs), new(xs))

    def test_prod(self):
        """
        Product p*p is correct.
        """
        pp = self.p*self.p
        npt.assert_array_almost_equal(self.p(xs)*self.p(xs),pp(xs))

    def test_square(self):
        def square(x):
            return self.p(x)*self.p(x)
        sq = Chebfun.from_function(square)
        npt.assert_array_less(0, sq(xs))
        self.sq = sq

    def test_chebyshev_points(self):
        """
        First and last interpolation points are -1 and 1
        """
        N = pow(2,5)
        pts = interpolation_points(N)
        npt.assert_array_almost_equal(pts[[0,-1]],np.array([1.,-1]))

    def test_N(self):
        """
        Check initialisation with a fixed N
        """
        N = len(self.p) - 1
        pN = Chebfun.from_function(f, N)
        self.assertEqual(len(pN.chebyshev_coefficients()), N+1)
        self.assertEqual(len(pN.chebyshev_coefficients()),len(pN))
        npt.assert_array_almost_equal(pN(xs), self.p(xs))
        npt.assert_array_almost_equal(pN.chebyshev_coefficients(),self.p.chebyshev_coefficients())

    def test_nonzero(self):
        """
        nonzero is True for Chebfun(f) and False for Chebfun(0)
        """
        self.assertTrue(self.p)
        mp = Chebfun.from_function(Zero)
        self.assertFalse(mp)


    def test_integrate(self):
        q = self.p.integrate()

    def test_differentiate(self):
        """
        Derivative of Chebfun(f) is close to Chebfun(derivative of f)
        """
        computed = self.p.differentiate()
        expected = fd
        npt.assert_allclose(computed(xs), expected(xs),)

    def test_interp_values(self):
        """
        Instanciate Chebfun from interpolation values.
        """
        p2 = Chebfun(self.p.values())
        npt.assert_almost_equal(self.p.chebyshev_coefficients(), p2.chebyshev_coefficients())
        npt.assert_array_almost_equal(self.p(xs), p2(xs))

    def test_equal(self):
        """
        Chebfun(f) is equal to itself.
        """
        self.assertEqual(self.p, chebfun(self.p))

class TestDifferentiate(unittest.TestCase):
    def test_diffquad(self):
        """
        Derivative of Chebfun(x**2/2) is close to identity function
        """
        self.p = .5*Chebfun.from_function(Quad)
        X = self.p.differentiate()
        npt.assert_array_almost_equal(X(xs), xs)

    def test_diff_x(self):
        """
        First and second derivative of Chebfun(x) are close to one and zero respectively.
        """
        self.p = Chebfun.from_function(Identity)
        one = self.p.differentiate()
        zero = one.differentiate()
        npt.assert_allclose(one(xs), 1.)
        npt.assert_allclose(Zero(xs), 0.)

    def test_diff_one(self):
        """
        Derivative of Chebfun(1) close to zero
        """
        one = Chebfun(1.)
        zero = one.differentiate()
        npt.assert_array_almost_equal(Zero(xs), 0.)

    def test_highdiff(self):
        """
        Higher order derivatives of exp(x)
        """
        e = Chebfun.from_function(lambda x:np.exp(x))
        e4 = e.differentiate(4)
        result = e4(xs)
        expected = e(xs)
        npt.assert_allclose(result, expected)
    
    def test_integrate(self):
        """
        Integrate exp
        """
        e = Chebfun.from_function(lambda x:np.exp(x))
        result = e.integrate()
        expected = e - 1
        npt.assert_allclose(result(xs), expected(xs))


class TestSimple(unittest.TestCase):
    def test_sum(self):
        """
        Integral of chebfun of x**2 on [-1,1] is 2/3
        """
        p = Chebfun.from_function(Quad)
        i = p.sum()
        npt.assert_array_almost_equal(i,2/3)

    def test_norm(self):
        """
        Norm of x**2 is sqrt(2/5)
        """
        p = Chebfun.from_function(Quad)
        self.assertAlmostEqual(p.norm(), np.sqrt(2/5))

    def test_dot(self):
        """
        f.0 = 0
        f.1 = f.sum()
        """
        p = Chebfun.from_function(np.sin)
        z = p.dot(Chebfun(0.))
        self.assertAlmostEqual(z, 0.)
        s = p.dot(Chebfun(1.))
        self.assertAlmostEqual(s, p.sum())

    def test_zero(self):
        """
        Chebfun for zero has the minimal degree 5
        """
        p = Chebfun.from_function(Zero)
        self.assertEqual(len(p),1) # should be equal to the minimum length, 1

    def test_repr(self):
        """
        Repr shows the interpolation values.
        """
        p = basis(1)
        s = repr(p)
        expected = '<Chebfun(array([ 1., -1.]))>'
        self.assertEqual(s, expected)

    def test_root(self):
        r = np.random.rand()
        p = Chebfun.from_function(lambda x: np.sin(x-r))
        roots = p.roots()
        npt.assert_allclose(roots, r)

class TestPolyfitShape(unittest.TestCase):
    def test_scalar(self):
        for datalen in [1,3]:
            coeffs = chebpolyfit(np.ones([datalen]))
            self.assertEqual(len(coeffs.shape), 1)

    def test_vector(self):
        for datalen in [1,3]:
            coeffs = chebpolyfit(np.ones([datalen, 2]))
            self.assertEqual(len(coeffs.shape), 2)

    def test_list(self):
        data = [[1.,2], [3,4]]
        adata = np.array(data)
        result = chebpolyfit(data)
        expected = chebpolyfit(adata)
        npt.assert_array_almost_equal(result, expected)

class TestEven(unittest.TestCase):
    def test_scalar(self):
        data = np.arange(5) # [0, 1, 2, 3, 4]
        result = even_data(data)
        expected = np.array(range(5) + range(1,4)[::-1]) # [0, 1, 2, 3, 4, 3, 2, 1]
        npt.assert_array_almost_equal(result, expected)

    def test_vector(self):
        data = np.array([[1.,2],[3.,4],[5,6]])
        result = even_data(data)
        expected = np.array([[1.,2],[3.,4],[5,6],[3.,4]])
        npt.assert_array_almost_equal(result, expected)

class TestDifferentiator(unittest.TestCase):
    def test_scalar_shape(self):
        """
        Differentiator returns the right shape
        """
        d = differentiator(np.array([1.]))
        self.assertEqual(np.shape(d), np.shape(np.array([0.])))

class TestInitialise(unittest.TestCase):
    def test_intlist(self):
        """
        Initialise with a list of integers
        """
        c = Chebfun([1,2,3])

    def test_chebcoefflist(self):
        """
        Initialise with a chebcoeff list
        """
        c = Chebfun.from_chebcoeff([1.,2.])

    def test_prune(self):
        """
        Prune works even if the coefficient is zero
        """
        N = Chebfun._cutoff(np.array([0.]), scale=1)
        self.assertEqual(N, 1)

def compare_ufunc(ufunc):
    # transformation from [-1, 1] to [1/4, 3/4]
    trans = lambda x: (x+2)/4
    x2 = Chebfun.from_function(trans)
    cf = ufunc(x2)
    result = cf.values()
    expected = ufunc(trans(cf.p.xi))
    npt.assert_allclose(result, expected)


class TestUfunc(unittest.TestCase):
    """
    Check that ufuncs work and give the right result.
    arccosh is not tested
    """

def _add_ufunc_test(ufunc):
    name = ufunc.__name__
    def test_func(self):
        compare_ufunc(ufunc)
    test_name = 'test_{}'.format(name)
    test_func.__name__ = test_name
    setattr(TestUfunc, test_name, test_func)

for func in [np.arccos, np.arcsin, np.arcsinh, np.arctan, np.arctanh, np.cos, np.sin, np.tan, np.cosh, np.sinh, np.tanh, np.exp, np.exp2, np.expm1, np.log, np.log2, np.log1p, np.sqrt, np.ceil, np.trunc, np.fabs, np.floor, np.abs]:
    _add_ufunc_test(func)

class Test_Misc(unittest.TestCase):
    def test_init_from_data(self):
        data = np.array([-1, 1.])
        c = Chebfun(data)

    def test_scalar_init(self):
        c = Chebfun(0.)
        npt.assert_allclose(c(xs), 0.)

    def test_empty_init(self):
        c = Chebfun()
        npt.assert_allclose(c(xs), 0.)

    def test_chebcoeff_one(self):
        c = Chebfun.from_chebcoeff(np.array([[1.],]))
        npt.assert_allclose(c(xs), 1.)

    def test_init_from_segment(self):
        c = Chebfun.from_function(segment)

    def test_init_from_circle(self):
        c = Chebfun.from_function(circle)

    def test_has_p(self):
        c1 = Chebfun.from_function(f, N=10)
        len(c1)
        c2 = Chebfun.from_function(f, )
        len(c2)

    def test_truncate(self, N=17):
        """
        Check that the Chebyshev coefficients are properly truncated.
        """
        small = Chebfun.from_function(f, N=N)
        new = Chebfun.from_function(small)
        self.assertEqual(len(new), len(small),)

    def test_vectorized(self):
        fv = np.vectorize(f)
        p = Chebfun.from_function(fv)

    def test_basis(self, ns=[0,5]):
        for n in ns:
            c = basis(n)
            npt.assert_array_almost_equal(c.chebyshev_coefficients(), np.array([0]*n+[1.]))

    def test_list_init(self):
        c = Chebfun([1.])
        npt.assert_array_almost_equal(c.chebyshev_coefficients(),np.array([1.]))

    def test_scalar_init(self):
        one = Chebfun(1.)
        npt.assert_array_almost_equal(one(xs), 1.)

    def test_no_convergence(self):
        with self.assertRaises(Chebfun.NoConvergence):
            Chebfun.from_function(np.sign)

    def test_runge(self):
        """
        Test some of the capabilities of operator overloading.
        """
        r = Chebfun.from_function(runge)
        x = basis(1)
        rr = 1./(1+25*x**2)
        npt.assert_almost_equal(r(xs),rr(xs), decimal=13)

    def test_idct(self, N=64):
        data = np.random.rand(N-1, 2)
        computed = chebpolyval(dct(data))
        npt.assert_allclose(computed, data[:N//2])

    def test_even_data(self):
        """
        even_data on vector of length N+1 returns a vector of size 2*N
        """
        N = 32
        data = np.random.rand(N+1).reshape(-1,1)
        even = even_data(data)
        self.assertEqual(len(even), 2*N)

    def test_chebpolyfit(self):
        N = 32
        data = np.random.rand(N-1, 2)
        coeffs = chebpolyfit(data)
        result = chebpolyval(coeffs)
        npt.assert_allclose(data, result)



    def test_underflow(self):
        self.skipTest('mysterious underflow error')
        p = Chebfun.from_function(piecewise_continuous, N=pow(2,10)-1)

class Test_Arithmetic(unittest.TestCase):
    def setUp(self):
        self.p1 = Chebfun.from_function(f)
        self.p2 = Chebfun.from_function(runge)

    def test_add(self):
        s = Chebfun.from_function(np.sin)
        c = Chebfun.from_function(np.cos)
        r = c + s
        def expected(x):
            return np.sin(x) + np.cos(x)
        npt.assert_allclose(r(xs), expected(xs))

    def test_scalar_mul(self):
        self.assertEqual(self.p1, self.p1)
        self.assertEqual(self.p1*1, 1*self.p1)
        self.assertEqual(self.p1*1, self.p1)
        self.assertEqual(0*self.p1, Chebfun.from_function(Zero))

    def test_commutativity(self):
        self.assertEqual(self.p1*self.p2, self.p2*self.p1)
        self.assertEqual(self.p1+self.p2, self.p2+self.p1)

    def test_minus(self):
        a = self.p1 - self.p2
        b = self.p2 - self.p1
        self.assertEqual(a+b,0)

    def test_cancel(self):
        """
        The Chebfun f-f should be equal to zero and of length one.
        """
        rm = -self.p2
        z = self.p2 + rm
        npt.assert_allclose(z(xs), np.zeros_like(xs), rtol=1e-7, atol=1e-8)
        self.assertEqual(len(z), 1)

    def test_add_mistype(self):
        """
        Possible to add a Chebfun and a function 
        """
        self.skipTest('not possible to add function and chebfun yet')
        def f(x):
            return np.sin(x)
        c = Chebfun.from_function(f)
        c + f


## class Test_2D(Test_Chebfun):
## 	def setUp(self):
## 		Chebfun.record = True
## 		self.p = Chebfun(segment,)

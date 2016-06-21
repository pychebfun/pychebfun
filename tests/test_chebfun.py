#!/usr/bin/env python
# coding: UTF-8

from __future__ import division
import os
import sys

import unittest
import numpy as np
import numpy.testing as npt

import pychebfun
from pychebfun import Chebfun, chebfun
from . import tools

import pytest

np.seterr(all='raise')

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
        self.fun = tools.circle

def Quad(x):
    return x*x

def piecewise_continuous(x):
    """
    The function is on the verge of being discontinuous at many points
    """
    return np.exp(x)*np.sin(3*x)*np.tanh(5*np.cos(30*x))

def runge(x):
    return 1./(1+25*x**2)


class Test_chebfuninit(unittest.TestCase):
    """
    Test that the initialisation function chebfun works as expected.
    """
    def test_from_function(self):
        cr = chebfun(tools.f)
        ce = Chebfun.from_function(tools.f)
        tools.assert_close(cr, ce)

    def test_from_chebcoeffs(self):
        coeffs = np.random.randn(10)
        cr = chebfun(chebcoeff=coeffs)
        ce = Chebfun.from_coeff(coeffs)
        tools.assert_close(cr, ce)

    def test_from_chebfun(self):
        ce = Chebfun.from_function(tools.f)
        cr = chebfun(ce)
        tools.assert_close(cr, ce)

    def test_from_values(self):
        values = np.random.randn(10)
        cr = chebfun(values)
        ce = Chebfun.from_data(values)
        tools.assert_close(cr, ce)

    def test_from_scalar(self):
        val = np.random.rand()
        cr = chebfun(val)
        ce = Chebfun.from_data([val])
        tools.assert_close(cr, ce)

    def test_error(self):
        """
        Error if chebfun is called with another type.
        """
        class C(object):
            pass
        with self.assertRaises(TypeError):
            chebfun(C())


class Test_sinsinexp(unittest.TestCase):
    """
    Tests with function np.sin(6*x) + np.sin(30*np.exp(x))
    """
    def setUp(self):
        # Construct the O(dx^-16) "spectrally accurate" chebfun p
        self.p = Chebfun.from_function(tools.f)

    def test_biglen(self):
        self.assertGreaterEqual(self.p.size(), 4)

    def test_len(self):
        """
        Length of chebfun is equal to the number of Cheb coefficients (i.e., degree)
        """
        self.assertEqual(self.p.size(), len(self.p.coefficients()))

    def test_error(self):
        """
        Chebfun is closed to function f up to tolerance
        """
        tools.assert_close(self.p, tools.f, atol=1e-13)

    def test_root(self):
        """
        Roots are zeros of the chebfun.
        """
        roots = self.p.roots()
        npt.assert_array_almost_equal(tools.f(roots),0)

    def test_all_roots(self):
        """
        Capture all rots.
        """
        roots = self.p.roots()
        self.assertEqual(len(roots),22)

    def test_chebcoeff(self):
        new = Chebfun.from_coeff(self.p.coefficients())
        tools.assert_close(self.p, new)

    def test_prod(self):
        """
        Product p*p is correct.
        """
        pp = self.p*self.p
        tools.assert_close(lambda x: self.p(x)*self.p(x), pp, atol=1e-13)

    def test_square(self):
        def square(x):
            return self.p(x)*self.p(x)
        sq = Chebfun.from_function(square)
        npt.assert_array_less(0, sq(tools.xs))
        self.sq = sq

    def test_chebyshev_points(self):
        """
        First and last interpolation points are -1 and 1
        """
        N = pow(2,5)
        pts = Chebfun.interpolation_points(N)
        npt.assert_array_almost_equal(pts[[0,-1]],np.array([1.,-1]))

    def test_N(self):
        """
        Check initialisation with a fixed N
        """
        N = self.p.size() - 1
        pN = Chebfun.from_function(tools.f, N=N)
        self.assertEqual(len(pN.coefficients()), N+1)
        self.assertEqual(len(pN.coefficients()),pN.size())
        tools.assert_close(pN, self.p)
        npt.assert_allclose(pN.coefficients(),self.p.coefficients())

    def test_nonzero(self):
        """
        nonzero is True for Chebfun(f) and False for Chebfun(0)
        """
        self.assertTrue(self.p)
        mp = Chebfun.from_function(tools.Zero)
        self.assertFalse(mp)


    def test_integrate(self):
        q = self.p.integrate()

    def test_differentiate(self):
        """
        Derivative of Chebfun(f) is close to Chebfun(derivative of f)
        """
        computed = self.p.differentiate()
        expected = tools.fd
        tools.assert_close(computed, expected)

    def test_interp_values(self):
        """
        Instanciate Chebfun from interpolation values.
        """
        p2 = Chebfun(self.p.values())
        npt.assert_almost_equal(self.p.coefficients(), p2.coefficients())
        tools.assert_close(self.p, p2)

    def test_equal(self):
        """
        Chebfun(f) is equal to itself.
        """
        tools.assert_close(self.p, Chebfun.from_function(self.p))

class TestDifferentiate(unittest.TestCase):
    def test_diffquad(self):
        """
        Derivative of Chebfun(x**2/2) is close to identity function
        """
        self.p = .5*Chebfun.from_function(Quad)
        X = self.p.differentiate()
        tools.assert_close(X, lambda x:x)

    def test_diff_x(self):
        """
        First and second derivative of Chebfun(x) are close to one and zero respectively.
        """
        self.p = Chebfun.from_function(tools.Identity)
        one = self.p.differentiate()
        zero = one.differentiate()
        npt.assert_allclose(one(tools.xs), 1.)
        npt.assert_allclose(tools.Zero(tools.xs), 0.)

    def test_diff_one(self):
        """
        Derivative of Chebfun(1) close to zero
        """
        one = Chebfun(1.)
        zero = one.differentiate()
        npt.assert_allclose(tools.Zero(tools.xs), 0.)

    def test_highdiff(self):
        """
        Higher order derivatives of exp(x)
        """
        e = Chebfun.from_function(lambda x:np.exp(x))
        e4 = e.differentiate(4)
        tools.assert_close(e4, e)

    def test_integrate(self):
        """
        Integrate exp
        """
        e = Chebfun.from_function(lambda x:np.exp(x))
        antideriv = e.integrate()
        result = antideriv - antideriv(antideriv._domain[0])
        tools.assert_close(result, e - e(antideriv._domain[0]))


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
        p = Chebfun.from_function(tools.Zero)
        self.assertEqual(p.size(),1) # should be equal to the minimum length, 1

    def test_repr(self):
        """
        Repr shows the interpolation values.
        """
        self.skipTest('Representation changed to include domain information')
        p = Chebfun.basis(1)
        s = repr(p)
        expected = '<Chebfun(array([ 1., -1.]))>'
        self.assertEqual(s, expected)

    def test_root(self):
        r = np.random.rand()
        p = Chebfun.from_function(lambda x: np.sin(x-r))
        roots = p.roots()
        npt.assert_allclose(roots, r)

    def test_basis(self, n=4):
        """
        Tn(cos(t)) = cos(nt)
        """
        Tn = Chebfun.basis(n)
        ts = np.linspace(0, 2*np.pi, 100)
        npt.assert_allclose(Tn(np.cos(ts)), np.cos(n*ts))

    def test_complex(self):
        n = 10
        r = np.random.randn(n) + 1j*np.random.randn(n)
        c = Chebfun.from_data(r)
        xs = Chebfun.interpolation_points(n)
        npt.assert_allclose(c(xs), r)

    def test_mx(self):
        c = Chebfun.from_function(lambda x:-x)
        tools.assert_close(c, lambda x:-x)

    def test_identity(self):
        c = Chebfun.identity()
        tools.assert_close(c, lambda x:x)

    @unittest.skip("real and imag do not work on chebfuns yet")
    def test_real_imag(self):
        datar = np.random.rand(10)
        datai = np.random.rand(10)
        cc = Chebfun.from_data(datar + 1j*datai)
        cr = Chebfun.from_data(datar)
        ci = Chebfun.from_data(datai)
        tools.assert_close(np.real(cc), cr)
        tools.assert_close(np.imag(cc), ci)


class TestPolyfitShape(unittest.TestCase):
    def test_scalar(self):
        for datalen in [1,3]:
            coeffs = Chebfun.polyfit(np.ones([datalen]))
            self.assertEqual(len(coeffs.shape), 1)

    def test_vector(self):
        for datalen in [1,3]:
            coeffs = Chebfun.polyfit(np.ones([datalen, 2]))
            self.assertEqual(len(coeffs.shape), 2)

    def test_list(self):
        data = [[1.,2], [3,4]]
        adata = np.array(data)
        result = Chebfun.polyfit(data)
        expected = Chebfun.polyfit(adata)
        npt.assert_array_almost_equal(result, expected)

class TestEven(unittest.TestCase):
    def test_scalar(self):
        data = np.arange(5) # [0, 1, 2, 3, 4]
        result = pychebfun.even_data(data)
        expected = np.array(list(range(5)) + list(range(1,4))[::-1]) # [0, 1, 2, 3, 4, 3, 2, 1]
        npt.assert_array_almost_equal(result, expected)

    def test_vector(self):
        data = np.array([[1.,2],[3.,4],[5,6]])
        result = pychebfun.even_data(data)
        expected = np.array([[1.,2],[3.,4],[5,6],[3.,4]])
        npt.assert_array_almost_equal(result, expected)

class TestDifferentiator(unittest.TestCase):
    def test_scalar_shape(self):
        """
        Differentiator returns the right shape
        """
        d = Chebfun.differentiator(np.array([1.]))
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
        c = Chebfun.from_coeff([1.,2.])

    def test_cutoff(self):
        """
        Prune works even if the coefficient is zero
        """
        N = Chebfun._cutoff(np.array([0.]), vscale=1)
        self.assertEqual(N, 1)

    def test_prune(self):
        N = 10
        coeffs = np.array([1.]+N*[0])
        c0 = Chebfun.from_coeff(coeffs)
        npt.assert_allclose(c0.coefficients(), [1.])
        c1 = Chebfun.from_coeff(coeffs, prune=False)
        npt.assert_allclose(c1.coefficients(), coeffs)




@pytest.mark.parametrize("ufunc", tools.ufunc_list, ids=tools.name_func)
def test_ufunc(ufunc):
    """
    Check that ufuncs work and give the right result.
    arccosh is not tested
    """
    # transformation from [-1, 1] to [1/4, 3/4]
    trans = lambda x: (x+2)/4
    x2 = Chebfun.from_function(trans)
    cf = ufunc(x2)
    assert isinstance(cf, Chebfun)
    result = cf.values()
    expected = ufunc(trans(cf.p.xi))
    npt.assert_allclose(result, expected)


class Test_Misc(unittest.TestCase):
    def test_init_from_data(self):
        data = np.array([-1, 1.])
        c = Chebfun(data)

    def test_scalar_init_zero(self):
        c = Chebfun(0.)
        npt.assert_allclose(c(tools.xs), 0.)

    def test_scalar_init_one(self):
        one = Chebfun(1.)
        npt.assert_array_almost_equal(one(tools.xs), 1.)

    def test_empty_init(self):
        c = Chebfun()
        npt.assert_allclose(c(tools.xs), 0.)

    def test_chebcoeff_one(self):
        c = Chebfun.from_coeff(np.array([[1.],]))
        npt.assert_allclose(c(tools.xs), 1.)

    def test_init_from_segment(self):
        c = Chebfun.from_function(segment)

    def test_init_from_circle(self):
        c = Chebfun.from_function(tools.circle)

    def test_has_p(self):
        c1 = Chebfun.from_function(tools.f, N=10)
        self.assertTrue(hasattr(c1, 'p'))
        c2 = Chebfun.from_function(tools.f, )
        self.assertTrue(hasattr(c2, 'p'))

    def test_truncate(self, N=17):
        """
        Check that the Chebyshev coefficients are properly truncated.
        """
        small = Chebfun.from_function(tools.f, N=N)
        new = Chebfun.from_function(small)
        self.assertEqual(new.size(), small.size(),)

    def test_vectorized(self):
        fv = np.vectorize(tools.f)
        p = Chebfun.from_function(fv)

    def test_basis(self, ns=[0,5]):
        for n in ns:
            c = Chebfun.basis(n)
            npt.assert_array_almost_equal(c.coefficients(), np.array([0]*n+[1.]))

    def test_list_init(self):
        c = Chebfun([1.])
        npt.assert_array_almost_equal(c.coefficients(),np.array([1.]))

    def test_no_convergence(self):
        with self.assertRaises(Chebfun.NoConvergence):
            Chebfun.from_function(np.sign)

    def test_runge(self):
        """
        Test some of the capabilities of operator overloading.
        """
        r = Chebfun.from_function(runge)
        x = Chebfun.basis(1)
        rr = 1./(1+25*x**2)
        tools.assert_close(r, rr, rtol=1e-13)

    def test_chebpolyfitval(self, N=64):
        data = np.random.rand(N-1, 2)
        computed = Chebfun.polyval(Chebfun.polyfit(data))
        npt.assert_allclose(computed, data)

    def test_even_data(self):
        """
        even_data on vector of length N+1 returns a vector of size 2*N
        """
        N = 32
        data = np.random.rand(N+1).reshape(-1,1)
        even = pychebfun.even_data(data)
        self.assertEqual(len(even), 2*N)

    def test_chebpolyfit(self):
        N = 32
        data = np.random.rand(N-1, 2)
        coeffs = Chebfun.polyfit(data)
        result = Chebfun.polyval(coeffs)
        npt.assert_allclose(data, result)

    def test_underflow(self):
        self.skipTest('mysterious underflow error')
        p = Chebfun.from_function(piecewise_continuous, N=pow(2,10)-1)

class Test_Arithmetic(unittest.TestCase):
    def setUp(self):
        self.p1 = Chebfun.from_function(tools.f)
        self.p2 = Chebfun.from_function(runge)

    def test_add(self):
        s = Chebfun.from_function(np.sin)
        c = Chebfun.from_function(np.cos)
        r = c + s
        def expected(x):
            return np.sin(x) + np.cos(x)
        tools.assert_close(r, expected)

    def test_scalar_mul(self):
        self.assertEqual(self.p1, self.p1)
        self.assertEqual(self.p1*1, 1*self.p1)
        self.assertEqual(self.p1*1, self.p1)
        self.assertEqual(0*self.p1, Chebfun.from_function(tools.Zero))

    def test_scalar(self):
        self.assertEqual(-self.p1, 0 - self.p1)
        self.assertEqual((1 - self.p1) - 1, -self.p1)

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
        npt.assert_allclose(z(tools.xs), np.zeros_like(tools.xs), rtol=1e-7, atol=1e-8)
        self.assertEqual(z.size(), 1)

    def test_add_mistype(self):
        """
        Possible to add a Chebfun and a function
        """
        self.skipTest('not possible to add function and chebfun yet')
        def f(x):
            return np.sin(x)
        c = Chebfun.from_function(f)
        result = c + f
        self.assertIsInstance(result, Chebfun)

    def test_equal(self):
        self.assertEqual(self.p1, self.p1)
        self.assertNotEqual(self.p1, self.p2)

class TestVector(unittest.TestCase):
    """
    Tests for the vector chebfuns.
    """

    def test_scalarvectormult(self):
        """
        Possible to multiply scalar with vector chebfun.
        """
        v = Chebfun.from_function(segment)
        s = np.sin(Chebfun.identity())
        m = s * v
        tools.assert_close(m[0], s*v[0])

    def test_slice(self):
        """
        Test slicing: f[0] should return the first component.
        """
        s = Chebfun.from_function(segment)
        tools.assert_close(s[0], Chebfun.identity())
        tools.assert_close(s[1], Chebfun(0.))
        tools.assert_close(s[:], s)

from .data import flat_chebfun_vals

class TestRoots(unittest.TestCase):
    """
    General root-finding tests.
    """

    def test_roots_of_flat_function(self):
        """
        Check roots() does not fail for extremely flat Chebfuns such
        as those representing cumulative distribution functions.
        """
        cdf = Chebfun.from_data(flat_chebfun_vals, domain=[-0.7, 0.7])
        npt.assert_allclose((cdf-0.05).roots(), 0.1751682246791747)

# class Test_2D(Test_Chebfun):
# 	def setUp(self):
# 		Chebfun.record = True
# 		self.p = Chebfun(segment,)

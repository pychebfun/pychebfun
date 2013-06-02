#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

# to prevent plots from popping up
import matplotlib
matplotlib.use('agg')

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

def segment(x):
    ones = np.ones_like(x)
    zeros = np.zeros_like(x)
    return np.vstack([ones, zeros])

def circle(x):
    return np.vstack([np.cos(np.pi*x), np.sin(np.pi*x)])

def f(x):
    return np.sin(6*x) + np.sin(30*np.exp(x))

def fd(x):
    """
    Derivative of f
    """
    return 6*np.cos(6*x) + np.cos(30*np.exp(x))*30*np.exp(x)

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
        Chebfun.record = True
        self.p = Chebfun(f,)

    def test_biglen(self):
        self.assertGreaterEqual(len(self.p), 4)

    def test_len(self):
        self.assertEqual(len(self.p), len(self.p.chebyshev_coefficients()))

    def test_error(self):
        x = xs
        err = abs(f(x)-self.p(x))
        npt.assert_array_almost_equal(self.p(x),f(x),decimal=13)

    def test_root(self):
        roots = self.p.roots()
        npt.assert_array_almost_equal(f(roots),0)

    def test_all_roots(self):
        roots = self.p.roots()
        self.assertEqual(len(roots),22)

    def test_chebcoeff(self):
        new = Chebfun(chebcoeff=self.p.ai)
        npt.assert_allclose(self.p(xs).reshape(-1,1), new(xs))

    def test_prod(self):
        pp = self.p*self.p
        npt.assert_array_almost_equal(self.p(xs)*self.p(xs),pp(xs))

    def test_square(self):
        def square(x):
            return self.p(x)*self.p(x)
        sq = Chebfun(square)
        npt.assert_array_less(0, sq(xs))
        self.sq = sq

    def test_chebyshev_points(self):
        N = pow(2,5)
        pts = interpolation_points(N)
        npt.assert_array_almost_equal(pts[[0,-1]],np.array([1.,-1]))

    def test_N(self):
        N = len(self.p) - 1
        pN = Chebfun(f, N)
        self.assertEqual(len(pN.chebyshev_coefficients()), N+1)
        self.assertEqual(len(pN.chebyshev_coefficients()),len(pN))
        npt.assert_array_almost_equal(pN(xs), self.p(xs))
        npt.assert_array_almost_equal(pN.chebyshev_coefficients(),self.p.chebyshev_coefficients())

    def test_record(self):
        p = Chebfun(f)
        self.assertEqual(len(p.bnds), 6)

    def test_zero(self):
        p = Chebfun(Zero)
        self.assertEqual(len(p),5) # should be equal to the minimum length, 4+1


    def test_nonzero(self):
        self.assertTrue(self.p)
        mp = Chebfun(Zero)
        self.assertFalse(mp)

    def test_integral(self):
        p = Chebfun(Quad)
        i = p.integral()
        npt.assert_array_almost_equal(i,2/3)

    def test_integrate(self):
        self.skipTest('bug in Chebfun.integrate')
        q = self.p.integrate()

    def test_differentiate(self):
        computed = self.p.differentiate()
        expected = Chebfun(fd)
        npt.assert_allclose(computed(xs), expected(xs).reshape(-1,1),)

    def test_diffquad(self):
        self.p = .5*Chebfun(Quad)
        X = self.p.differentiate()
        npt.assert_array_almost_equal(X(xs), xs.reshape(-1,1))

    def test_diff_x(self):
        self.p = Chebfun(Identity)
        one = self.p.differentiate()
        zero = one.differentiate()
        npt.assert_allclose(one(xs), 1.)
        npt.assert_allclose(Zero(xs), 0.)

    def test_diff_one(self):
        one = Chebfun(1.)
        zero = one.differentiate()
        npt.assert_array_almost_equal(Zero(xs), 0.)

    def test_interp_values(self):
        """
        Instanciate Chebfun from interpolation values.
        """
        p2 = Chebfun(self.p.f)
        npt.assert_almost_equal(self.p.ai, p2.ai)
        npt.assert_array_almost_equal(self.p(xs), p2(xs))

    def test_equal(self):
        self.assertEqual(self.p, Chebfun(self.p))

class TestPlot(unittest.TestCase):
    def setUp(self):
        # Constuct the O(dx^-16) "spectrally accurate" chebfun p
        Chebfun.record = True
        self.p = Chebfun()
        self.p.init_from_function(f)

    def test_plot(self):
        self.p.plot()

    def test_plot_interpolation_points(self):
        plt.clf()
        self.p.plot()
        a = plt.gca()
        self.assertEqual(len(a.lines),2)
        plt.clf()
        self.p.plot(with_interpolation_points=False)
        a = plt.gca()
        self.assertEqual(len(a.lines),1)

    def test_cheb_plot(self):
        self.p.compare(f)

    def test_chebcoeffplot(self):
        self.p.chebcoeffplot()


class Test_Misc(unittest.TestCase):
    def test_init_from_data(self):
        data = np.array([-1, 1.])
        c = Chebfun(data)

    def test_empty_init(self):
        c = Chebfun()

    def test_chebcoeff_one(self):
        c = Chebfun(chebcoeff=np.array([[1.],]))
        npt.assert_array_almost_equal(c(xs), 1.)

    def test_init_from_vector_function(self):
        c = Chebfun(segment)

    def test_plot_circle(self):
        c = Chebfun(circle)
        c.plot()

    def test_has_p(self):
        c1 = Chebfun(f, N=10)
        len(c1)
        c2 = Chebfun(f, )
        len(c2)

    def test_truncate(self, N=17):
        """
        Check that the Chebyshev coefficients are properly truncated.
        """
        small = Chebfun(f, N=N)
        new = Chebfun(small)
        self.assertEqual(len(new), len(small),)

    def test_error(self):
        chebpolyplot(f)

    def test_vectorized(self):
        fv = np.vectorize(f)
        p = Chebfun(fv)

    def test_chebpoly(self, ns=[0,5]):
        for n in ns:
            c = chebpoly(n)
            npt.assert_array_almost_equal(c.chebyshev_coefficients(), np.array([0]*n+[1.]).reshape(-1,1))

    def test_list_init(self):
        c = Chebfun([1.])
        npt.assert_array_almost_equal(c.chebyshev_coefficients(),np.array([[1.]]))

    def test_scalar_init(self):
        one = Chebfun(1.)
        npt.assert_array_almost_equal(one(xs), 1.)

    def test_no_convergence(self):
        with self.assertRaises(Chebfun.NoConvergence):
            Chebfun(np.sign)

    def test_runge(self):
        """
        Test some of the capabilities of operator overloading.
        """
        r = Chebfun(runge)
        x = chebpoly(1)
        rr = 1./(1+25*x**2)
        npt.assert_almost_equal(r(xs),rr(xs), decimal=13)

    def test_idct(self, N=64):
        data = np.random.rand(N-1, 2)
        computed = idct(dct(data))
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
        result = idct(coeffs)
        npt.assert_allclose(data, result)



    def test_underflow(self):
        self.skipTest('mysterious underflow error')
        Chebfun.max_nb_dichotomy = 13
        p = Chebfun(piecewise_continuous)

class Test_Arithmetic(unittest.TestCase):
    def setUp(self):
        self.p1 = Chebfun(f)
        self.p2 = Chebfun(runge)

    def test_scalar_mul(self):
        self.assertEqual(self.p1, self.p1)
        self.assertEqual(self.p1*1, 1*self.p1)
        self.assertEqual(self.p1*1, self.p1)
        self.assertEqual(0*self.p1, Chebfun(Zero))

    def test_commutativity(self):
        self.assertEqual(self.p1*self.p2, self.p2*self.p1)
        self.assertEqual(self.p1+self.p2, self.p2+self.p1)

    def test_minus(self):
        a = self.p1 - self.p2
        b = self.p2 - self.p1
        self.assertEqual(a+b,0)

    def test_neg(self):
        self.skipTest('problem due to lack of scaling strategy')
        rm = -self.p2
        z = self.p2 + rm



if __name__ == '__main__':
    unittest.main()
    ## suite = unittest.TestLoader().loadTestsFromTestCase(Test_Chebfun)
    ## unittest.TextTestRunner(verbosity=2).run(suite)
    

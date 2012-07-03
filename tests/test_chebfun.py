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


def f(x):
    return np.sin(6*x) + np.sin(30*np.exp(x))


def fd(x):
    """
    Derivative of f
    """
    return 6*np.cos(6*x) + np.cos(30*np.exp(x))*30*np.exp(x)

def piecewise_continuous(x):
    """
    The function is on the verge of being discontinuous at many points
    """
    return np.exp(x)*np.sin(3*x)*np.tanh(5*np.cos(30*x))

def runge(x):
    return 1./(1+25*x**2)

@np.vectorize
def zero(x):
    return 0.

xs = np.linspace(-1,1,1000)

class Test_Chebfun(unittest.TestCase):
    def setUp(self):
        # Constuct the O(dx^-16) "spectrally accurate" chebfun p
        Chebfun.record = True
        self.p = Chebfun(f,)

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

    def test_plot(self):
        self.p.plot()

    def test_plot_interpolation_points(self):
        plt.clf()
        self.p.plot()
        a = plt.gca()
        self.assertEqual(len(a.lines),2)
        plt.clf()
        self.p.plot(interpolation_points=False)
        a = plt.gca()
        self.assertEqual(len(a.lines),1)

    def test_chebcoeff(self):
        new = Chebfun(chebcoeff=self.p.ai)
        npt.assert_allclose(self.p(xs), new(xs))

    def test_cheb_plot(self):
        self.p.compare(f)

    def test_chebcoeffplot(self):
        self.p.chebcoeffplot()

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
        pts = self.p.interpolation_points(N)
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
        self.assertEqual(len(p.bnds), 7)

    def test_zero(self):
        p = Chebfun(zero)
        self.assertEqual(len(p),5) # should be equal to the minimum length, 4+1


    def test_nonzero(self):
        self.assertTrue(self.p)
        mp = Chebfun(zero)
        self.assertFalse(mp)

    def test_integral(self):
        def q(x):
            return x*x
        p = Chebfun(q)
        i = p.integral()
        self.assertAlmostEqual(i,2/3)

    def test_differentiate(self):
        computed = self.p.differentiate()
        expected = Chebfun(fd)
        npt.assert_allclose(computed(xs), expected(xs),)

    def test_interp_values(self):
        """
        Instanciate Chebfun from interpolation values.
        """
        p2 = Chebfun(self.p.f)
        npt.assert_almost_equal(self.p.ai, p2.ai)
        npt.assert_array_almost_equal(self.p(xs), p2(xs))

    def test_equal(self):
        self.assertEqual(self.p, Chebfun(self.p))


class Test_Misc(unittest.TestCase):
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

    def test_examples(self):
        """
        Check that the examples can be executed.
        """
        here = os.path.dirname(__file__)
        example_folder = os.path.join(here,os.path.pardir,'examples')
        files = os.listdir(example_folder)
        for example in files:
            file_name = os.path.join(example_folder,example)
            try:
                execfile(file_name, {})
            except Exception as e:
                raise Exception('Error in {0}: {0}'.format(example), e)

    def test_chebpoly(self, ns=[0,5]):
        for n in ns:
            c = chebpoly(n)
            npt.assert_array_almost_equal(c.chebyshev_coefficients(), [0]*n+[1.])

    def test_list_init(self):
        c = Chebfun([1.])
        npt.assert_array_almost_equal(c.chebyshev_coefficients(),[1.])

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
        data = np.random.rand(N-1)
        computed = idct(dct(data))
        npt.assert_allclose(computed, data[:N//2])

    def test_even_data(self):
        """
        even_data on vector of length N+1 returns a vector of size 2*N
        """
        N = 32
        data = np.random.rand(N+1)
        even = even_data(data)
        self.assertEqual(len(even), 2*N)


    @unittest.expectedFailure
    def test_underflow(self):
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
        self.assertEqual(0*self.p1, zero)

    def test_commutativity(self):
        self.assertEqual(self.p1*self.p2, self.p2*self.p1)
        self.assertEqual(self.p1+self.p2, self.p2+self.p1)

    def test_minus(self):
        a = self.p1 - self.p2
        b = self.p2 - self.p1
        self.assertEqual(a+b,0)

    @unittest.expectedFailure
    @unittest.expectedFailure
    def test_neg(self):
        rm = -self.p2
        z = self.p2 + rm



if __name__ == '__main__':
    unittest.main()
    ## suite = unittest.TestLoader().loadTestsFromTestCase(Test_Chebfun)
    ## unittest.TextTestRunner(verbosity=2).run(suite)
    

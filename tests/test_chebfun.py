#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

# to prevent plots from popping up
import matplotlib
matplotlib.use('agg')

import os

from pychebfun import *

import numpy as np
np.seterr(all='raise')
import numpy.testing as npt
import nose.tools as nt


def f(x):
    return np.sin(6*x) + np.sin(30*np.exp(x))

@np.vectorize
def zero(x):
    return 0.

class Test_Chebfun(object):
    def setUp(self):
        # Constuct the O(dx^-16) "spectrally accurate" chebfun p
        Chebfun.record = True
        self.p = Chebfun(f,)
        self.xs = np.linspace(-1,1,1000)

    def test_len(self):
        nt.assert_equal(len(self.p), len(self.p.chebyshev_coefficients()))

    def test_error(self):
        x = self.xs
        err = abs(f(x)-self.p(x))
        npt.assert_array_almost_equal(self.p(x),f(x),decimal=13)

    def test_root(self):
        roots = self.p.roots()
        npt.assert_array_almost_equal(f(roots),0)

    def test_plot(self):
        self.p.plot()

    def test_cheb_plot(self):
        self.p.compare(f)

    def test_chebcoeffplot(self):
        self.p.chebcoeffplot()

    def test_prod(self):
        pp = self.p*self.p
        npt.assert_array_almost_equal(self.p(self.xs)*self.p(self.xs),pp(self.xs))

    def test_square(self):
        def square(x):
            return self.p(x)*self.p(x)
        sq = Chebfun(square)
        npt.assert_array_less(0, sq(self.xs))
        self.sq = sq
    
    def test_chebyshev_points(self):
        N = pow(2,5)
        pts = self.p.interpolation_points(N)
        npt.assert_array_almost_equal(pts[[0,-1]],np.array([1.,-1]))

    def test_even_data(self):
        """
        even_data on vector of length N+1 returns a vector of size 2*N
        """
        N = 32
        data = np.random.rand(N+1)
        even = self.p.even_data(data)
        nt.assert_equal(len(even), 2*N)

    def test_N(self):
        N = len(self.p) - 1
        pN = Chebfun(f, N)
        nt.assert_equal(len(pN.chebyshev_coefficients()), N+1)
        nt.assert_equal(len(pN.chebyshev_coefficients()),len(pN))
        npt.assert_array_almost_equal(pN(self.xs), self.p(self.xs))
        npt.assert_array_almost_equal(pN.chebyshev_coefficients(),self.p.chebyshev_coefficients())

    def test_record(self):
        p = Chebfun(f)
        nt.assert_equal(len(p.bnds), 7)

    def test_zero(self):
        p = Chebfun(zero)
        nt.assert_equal(len(p),5) # should be equal to the minimum length, 4+1


    def test_nonzero(self):
        nt.assert_true(self.p)
        mp = Chebfun(zero)
        nt.assert_false(mp)

    def test_integral(self):
        def q(x):
            return x*x
        p = Chebfun(q)
        i = p.integral()
        nt.assert_almost_equal(i,2/3)

    def test_interp_values(self):
        """
        Instianciate Chebfun from interpolation values.
        """
        p2 = Chebfun(self.p.f)
        npt.assert_almost_equal(self.p.ai, p2.ai)
        npt.assert_array_almost_equal(self.p(self.xs), p2(self.xs))

def test_truncate(N=17):
    """
    Check that the Chebyshev coefficients are properly truncated.
    """
    small = Chebfun(f, N=N)
    new = Chebfun(small)
    nt.assert_equal(len(new), len(small),)

def test_error():
    chebpolyplot(f)

def test_vectorized():
    fv = np.vectorize(f)
    p = Chebfun(fv)

def test_examples():
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

def test_chebpoly(ns=[0,5]):
    for n in ns:
        c = chebpoly(n)
        npt.assert_array_almost_equal(c.chebyshev_coefficients(), [0]*n+[1.])

def test_list_init():
    c = Chebfun([1.])
    npt.assert_array_almost_equal(c.chebyshev_coefficients(),[1.])

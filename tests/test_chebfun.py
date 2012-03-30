# to prevent plots from popping up
import matplotlib
matplotlib.use('agg')

from pychebfun import *

import numpy as np
import numpy.testing as npt
import nose.tools as nt


def f(x):
    return np.sin(6*x) + np.sin(30*np.exp(x))

class Test_Chebfun(object):
    def setUp(self):
        # Constuct the O(dx^-16) "spectrally accurate" chebfun p
        self.p = Chebfun(f)
        self.xs = np.linspace(-1,1,1000)
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

def test_truncate(N=17):
    """
    Check that the Chebyshev coefficients are properly truncated.
    """
    small = Chebfun(f, N=N)
    new = Chebfun(small)
    nt.assert_equal(new.N, small.N,)

def test_error():
    chebpolyplot(f)

def test_vectorized():
    fv = np.vectorize(f)
    p = Chebfun(fv)


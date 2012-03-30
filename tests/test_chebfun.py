# to prevent plots from popping up
import matplotlib
matplotlib.use('agg')

from pychebfun import *

import numpy as np
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
        pts = self.p.chebyshev_points(N)
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
        pN = Chebfun(f, len(self.p)+1)
        npt.assert_array_almost_equal(pN(self.xs), self.p(self.xs))

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


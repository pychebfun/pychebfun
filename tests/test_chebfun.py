# to prevent plots from popping up
import matplotlib
matplotlib.use('agg')

from pychebfun import *

import numpy as np
import numpy.testing as npt


def f(x):
    return np.sin(6*x) + np.sin(30*np.exp(x))

class Test_Chebfun(object):
    def setUp(self):
        # Constuct the O(dx^-16) "spectrally accurate" chebfun p
        self.p = Chebfun(f)
    def test_error(self):
        x = np.linspace(-1,1,1000)
        err = abs(f(x)-self.p(x))
        npt.assert_array_almost_equal(self.p(x),f(x))

    def test_root(self):
        roots = self.p.roots()
        npt.assert_array_almost_equal(f(roots),0)

    def test_cheb_plot(self):
        chebplot(f,self.p)

def test_error():
    chebpolyplot(f)

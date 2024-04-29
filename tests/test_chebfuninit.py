
from pychebfun import Chebfun
from pychebfun.chebfun_init import chebfun
from . import tools
import numpy as np
import unittest

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

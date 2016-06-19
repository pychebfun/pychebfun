#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

from pychebfun import Chebfun
import operator
import unittest
import pytest
from . import tools
import numpy as np
import numpy.testing as npt

#------------------------------------------------------------------------------
# Unit test for arbitrary interval Chebfuns
#------------------------------------------------------------------------------



class TestDomain(unittest.TestCase):
    def test_mismatch(self):
        c1 = Chebfun.identity()
        c2 = Chebfun.from_function(lambda x:x, domain=[2,3])
        for op in [operator.add, operator.sub, operator.mul, operator.truediv]:
            with self.assertRaises(Chebfun.DomainMismatch):
                op(c1, c2)

    def test_restrict(self):
        x = Chebfun.identity()
        with self.assertRaises(ValueError):
            x.restrict([-2,0])
        with self.assertRaises(ValueError):
            x.restrict([0,2])




@pytest.mark.parametrize("ufunc", tools.ufunc_list, ids=tools.name_func)
def test_init(ufunc):
    xx = Chebfun.from_function(lambda x: x,[0.25,0.75])
    ff = ufunc(xx)
    assert isinstance(ff, Chebfun)
    result = ff.values()
    expected = ufunc(ff._ui_to_ab(ff.p.xi))
    npt.assert_allclose(result, expected)


#------------------------------------------------------------------------------
# Test the restrict operator
#------------------------------------------------------------------------------

from . import data

@pytest.mark.parametrize('ff', [Chebfun.from_function(tools.f,[-3,4])])
@pytest.mark.parametrize('domain', data.IntervalTestData.domains)
def test_restrict(ff, domain):
    ff_ = ff.restrict(domain)
    xx = tools.map_ui_ab(tools.xs, domain[0],domain[1])
    tools.assert_close(tools.f, ff_, xx)

#------------------------------------------------------------------------------
# Add the arbitrary interval tests
#------------------------------------------------------------------------------

@pytest.fixture(params=list(range(5)))
def tdata(request):
    index = request.param
    class TData(): pass
    tdata = TData()
    tdata.function = data.IntervalTestData.functions[0]
    tdata.function_d = data.IntervalTestData.first_derivs[0]
    tdata.domain = data.IntervalTestData.domains[index]
    tdata.roots = data.IntervalTestData.roots[0][index]
    tdata.integral = data.IntervalTestData.integrals[0][index]
    tdata.chebfun = Chebfun.from_function(tdata.function, tdata.domain)
    return tdata


class TestArbitraryIntervals(object):
    """Test the various operations for Chebfun on arbitrary intervals"""

    def test_evaluation(self, tdata):
        xx = tools.map_ui_ab(tools.xs, tdata.domain[0], tdata.domain[1])
        tools.assert_close(tdata.chebfun, tdata.function, xx)

    def test_domain(self, tdata):
        assert tdata.chebfun._domain[0] == tdata.domain[0]
        assert tdata.chebfun._domain[1] == tdata.domain[1]

    def test_first_deriv(self, tdata):
        xx = tools.map_ui_ab(tools.xs, tdata.domain[0], tdata.domain[1])
        tools.assert_close(tdata.chebfun.differentiate(), tdata.function_d, xx)

    def test_definite_integral(self, tdata):
        actual = tdata.integral
        npt.assert_allclose(tdata.chebfun.sum(), actual, rtol=1e-12)

    def test_roots(self, tdata):
        actual = tdata.roots
        npt.assert_allclose(np.sort(tdata.chebfun.roots()), actual, rtol=1e-12)


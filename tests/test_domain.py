#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

from pychebfun import *
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




@pytest.mark.parametrize("ufunc", tools.ufunc_list)
def test_func(ufunc):
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
def test_func(ff, domain):
    ff_ = ff.restrict(domain)
    xx = tools.map_ui_ab(tools.xs, domain[0],domain[1])
    tools.assert_close(tools.f, ff_, xx)

#------------------------------------------------------------------------------
# Add the arbitrary interval tests
#------------------------------------------------------------------------------
class HarnessArbitraryIntervals(object):
    """Test the various operations for Chebfun on arbitrary intervals"""

    def test_domain(self):
        self.assertEqual(self.chebfun._domain[0],self.domain[0])
        self.assertEqual(self.chebfun._domain[1],self.domain[1])

    def test_evaluation(self):
        xx = tools.map_ui_ab(tools.xs, self.domain[0], self.domain[1])
        tools.assert_close(self.chebfun, self.function, xx)

    def test_first_deriv(self):
        xx = tools.map_ui_ab(tools.xs, self.domain[0], self.domain[1])
        tools.assert_close(self.chebfun.differentiate(), self.function_d, xx)

    def test_definite_integral(self):
        actual = self.integral
        self.assertAlmostEqual(self.chebfun.sum(), actual, places=14)

    def test_roots(self):
        actual = self.roots
        self.assertAlmostEqual(np.linalg.norm(np.sort(self.chebfun.roots()) - actual), 0., places=12)


def _get_class_name(template, f, domain_index):
    return template.format(f.__name__, domain_index)

def _get_setup(func, func_d, dom_data):
    def setUp(self):
        self.function = func
        self.function_d = func_d
        self.domain = dom_data["domain"]
        self.roots = dom_data["roots"]
        self.integral = dom_data["integral"]
        self.chebfun = Chebfun.from_function(self.function, self.domain)
    return setUp

global_dict = globals()
for fdata in data.interval_test_data:
    for index, dom_data in enumerate(fdata["domains"]):
        cls_name = _get_class_name("TestArbitraryInterval_{}_{}", fdata["function"], index)
        global_dict[cls_name] = type(cls_name, (HarnessArbitraryIntervals, unittest.TestCase), {"setUp": _get_setup(fdata["function"], fdata["function_d"], dom_data)})


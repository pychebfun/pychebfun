#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

from pychebfun import *
import unittest
from .tools import *
import numpy as np
import numpy.testing as npt

#------------------------------------------------------------------------------
# Unit test for arbitrary interval Chebfuns
#------------------------------------------------------------------------------


def _get_chebfun(f, domain):
    return Chebfun.from_function(f, domain)

def _get_class_name(template, f, domain_index):
    return template.format(f.__name__, domain_index)

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

class HarnessArbitraryIntervals(object):
    """Test the various operations for Chebfun on arbitrary intervals"""

    def test_domain(self):
        self.assertEqual(self.chebfun._domain[0],self.domain[0])
        self.assertEqual(self.chebfun._domain[1],self.domain[1])

    def test_evaluation(self):
        xx = map_ui_ab(xs, self.domain[0], self.domain[1])
        assert_close(self.chebfun, self.function, xx)

    def test_first_deriv(self):
        xx = map_ui_ab(xs, self.domain[0], self.domain[1])
        assert_close(self.chebfun.differentiate(), self.function_d, xx)

    def test_definite_integral(self):
        actual = self.integral
        self.assertAlmostEqual(self.chebfun.sum(), actual, places=14)

    def test_roots(self):
        actual = self.roots
        self.assertAlmostEqual(norm(np.sort(self.chebfun.roots()) - actual), 0., places=12)

def _get_setup(func, func_d, dom_data):
    def setUp(self):
        self.function = func
        self.function_d = func_d
        self.domain = dom_data["domain"]
        self.roots = dom_data["roots"]
        self.integral = dom_data["integral"]
        self.chebfun = _get_chebfun(self.function, self.domain)
    return setUp



class TestUfuncIntervals(unittest.TestCase):
    pass

def compare_ufunc_arb_interval(self, ufunc):
    xx = Chebfun.from_function(lambda x: x,[0.25,0.75])
    ff = ufunc(xx)
    self.assertIsInstance(ff, Chebfun)
    result = ff.values()
    expected = ufunc(ff._ui_to_ab(ff.p.xi))
    npt.assert_allclose(result, expected)

def _add_ufunc_test_arb_interval(ufunc):
    name = ufunc.__name__
    def test_func(self):
        compare_ufunc_arb_interval(self, ufunc)
    test_name = 'test_non_ui_{}'.format(name)
    test_func.__name__ = test_name
    setattr(TestUfuncIntervals, test_name, test_func)



for func in [np.arccos, np.arcsin, np.arcsinh, np.arctan, np.arctanh, np.cos, np.sin, np.tan, np.cosh, np.sinh, np.tanh, np.exp, np.exp2, np.expm1, np.log, np.log2, np.log1p, np.sqrt, np.ceil, np.trunc, np.fabs, np.floor, np.abs]:
    _add_ufunc_test_arb_interval(func)

#------------------------------------------------------------------------------
# Test the restrict operator
#------------------------------------------------------------------------------

def _add_test_restrict_method(domain, index):
    def test_func(self):
        ff = self.ff.restrict(domain)
        xx = map_ui_ab(xs, domain[0],domain[1])
        assert_close(f, ff, xx)
    test_name = 'test_restrict_method_dom{}'.format(index)
    test_func.__name__ = test_name
    setattr(TestRestrict, test_name, test_func)

class TestRestrict(unittest.TestCase):
    """Test the restrict operator"""
    def setUp(self):
        self.ff = Chebfun.from_function(f,[-3,4])

#------------------------------------------------------------------------------
# Test data
#------------------------------------------------------------------------------

class IntervalTestData(object):
    functions = [f]
    first_derivs = [fd]

    domains = [(1,2),(0,2),(-1,0),(-.2*np.pi,.2*np.e),(-1,1)]

    integrals = [
        [ 0.032346217980525,  0.030893429600387, -0.014887469493652,
         -0.033389463703032, -0.016340257873789, ]
    ]

    roots = [
        [
            np.array([
                1.004742754531498, 1.038773298601836, 1.073913103930722,
                1.115303578807479, 1.138876334576409, 1.186037005063195,
                1.200100773491540, 1.251812490296546, 1.257982114030372,
                1.312857486088040, 1.313296484543653, 1.365016316032836,
                1.371027655848883, 1.414708808202124, 1.425447888640173,
                1.462152640981920, 1.476924360913394, 1.507538306301423,
                1.525765627652155, 1.551033406767893, 1.572233571395834,
                1.592786143530423, 1.616552437657155, 1.632928169757349,
                1.658915772490721, 1.671576942342459, 1.699491823230094,
                1.708837673403015, 1.738427795274605, 1.744804960074507,
                1.775853245044121, 1.779564153811983, 1.811882812082608,
                1.813192517312102, 1.845760207165999, 1.846618439572035,
                1.877331112646444, 1.880151194495009, 1.907963575049332,
                1.912562771369236, 1.937711007329229, 1.943926743585850,
                1.966622430081970, 1.974309611716701, 1.994742937003962,
            ]),

            np.array([
                0.038699154393837, 0.170621357069026, 0.196642349303247,
                0.335710810755860, 0.360022217617733, 0.459687243605995,
                0.515107092342894, 0.571365105600701, 0.646902333813374,
                0.672854750953472, 0.761751991347867, 0.765783134619707,
                0.851427319155724, 0.863669737544800, 0.930805860269712,
                0.955368374256150,
                1.004742754531498, 1.038773298601836, 1.073913103930722,
                1.115303578807479, 1.138876334576409, 1.186037005063195,
                1.200100773491540, 1.251812490296546, 1.257982114030372,
                1.312857486088040, 1.313296484543653, 1.365016316032836,
                1.371027655848883, 1.414708808202124, 1.425447888640173,
                1.462152640981920, 1.476924360913394, 1.507538306301423,
                1.525765627652155, 1.551033406767893, 1.572233571395834,
                1.592786143530423, 1.616552437657155, 1.632928169757349,
                1.658915772490721, 1.671576942342459, 1.699491823230094,
                1.708837673403015, 1.738427795274605, 1.744804960074507,
                1.775853245044121, 1.779564153811983, 1.811882812082608,
                1.813192517312102, 1.845760207165999, 1.846618439572035,
                1.877331112646444, 1.880151194495009, 1.907963575049332,
                1.912562771369236, 1.937711007329229, 1.943926743585850,
                1.966622430081970, 1.974309611716701, 1.994742937003962,
            ]),

            np.array([
                -0.928510879374692, -0.613329324979852, -0.437747415493617,
                -0.357059979912156, -0.143371301774133, -0.075365172766102,
            ]),

            np.array([
                -0.613329324979852, -0.437747415493618, -0.357059979912156,
                -0.143371301774133, -0.075365172766103,  0.038699154393837,
                 0.170621357069026,  0.196642349303248,  0.335710810755860,
                 0.360022217617734,  0.459687243605995,  0.515107092342894,
            ]),

            np.array([
                -0.928510879374692, -0.613329324979852, -0.437747415493617,
                -0.357059979912156, -0.143371301774133, -0.075365172766102,
                 0.038699154393837,  0.170621357069026,  0.196642349303247,
                 0.335710810755860,  0.360022217617733,  0.459687243605995,
                 0.515107092342894,  0.571365105600701,  0.646902333813374,
                 0.672854750953472,  0.761751991347867,  0.765783134619707,
                 0.851427319155724,  0.863669737544800,  0.930805860269712,
                 0.955368374256150,
            ])
        ]
    ]

interval_test_data = [{"function": func, "function_d": func_d, "domains":[{"domain": domain, "integral":integral, "roots":roots} for (domain,integral,roots) in zip(IntervalTestData.domains, fints, froots)]} for (func, func_d, fints, froots) in zip(IntervalTestData.functions, IntervalTestData.first_derivs, IntervalTestData.integrals, IntervalTestData.roots)]

#------------------------------------------------------------------------------
# Add the arbitrary interval tests
#------------------------------------------------------------------------------

global_dict = globals()
for fdata in interval_test_data:
    for index, dom_data in enumerate(fdata["domains"]):
        cls_name = _get_class_name("TestArbitraryInterval_{}_{}", fdata["function"], index)
        global_dict[cls_name] = type(cls_name, (HarnessArbitraryIntervals, unittest.TestCase), {"setUp": _get_setup(fdata["function"], fdata["function_d"], dom_data)})
#------------------------------------------------------------------------------
# Test the restrict operator
#------------------------------------------------------------------------------

for index, domain in enumerate(IntervalTestData.domains):
    _add_test_restrict_method(domain, index)

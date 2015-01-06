#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

from pychebfun import *

import numpy as np
np.seterr(all='raise')
import numpy.testing as npt

import unittest

from tools import *

class TestPlot(unittest.TestCase):
    def setUp(self):
        # Constuct the O(dx^-16) "spectrally accurate" chebfun p
        self.p = Chebfun.from_function(f)

    def test_plot(self):
        xs,ys,xi,yi,d = self.p.plot_data()
        self.assertEqual(d, 1)
        npt.assert_allclose(ys, f(xs))
        self.p.plot()

    def test_plot_interpolation_points(self):
        plt.clf()
        self.p.plot(with_interpolation_points=True)
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

    def test_plot_circle(self):
        T = .5
        def cirper(x):
            return circle(x, period=T)
        c = Chebfun.from_function(cirper, domain=[0,T])
        xs,ys,xi,yi,d = c.plot_data()
        self.assertEqual(d, 2,)
        for X,Y in [(xs,ys), (xi,yi)]:
            dist = np.square(X) + np.square(Y)
            npt.assert_allclose(dist, 1, err_msg="The plot should be a circle")
        c.plot()

    def test_plot_complex(self):
        c = np.exp(1j*Chebfun.identity(domain=[-np.pi,np.pi]))
        xs,ys,xi,yi,d = c.plot_data()
        self.assertEqual(d, 2, "dimension is two for complex chebfun")
        for X,Y in [(xs,ys), (xi,yi)]:
            dist = np.square(X) + np.square(Y)
            npt.assert_allclose(dist, 1, err_msg="The plot should be a circle")
        c.plot()

    def test_error(self):
        chebpolyplot(self.p)



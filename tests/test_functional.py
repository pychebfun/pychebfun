#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

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

from tools import *

class TestFunctional(unittest.TestCase):
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
        c = Chebfun.from_function(circle)
        xs,ys,xi,yi,d = c.plot_data()
        self.assertEqual(d, 2,)
        dist = np.square(xs) + np.square(ys)
        npt.assert_allclose(dist, 1, err_msg="The plot should be a circle")
        c.plot()

    def test_plot_complex(self):
        c = np.exp(1j*np.pi*Chebfun.identity())
        xs,ys,xi,yi,d = c.plot_data()
        self.assertEqual(d, 2, "dimension is two for complex chebfun")
        dist = np.square(xs) + np.square(ys)
        npt.assert_allclose(dist, 1, err_msg="The plot should be a circle")
        c.plot()

    def test_error(self):
        chebpolyplot(self.p)



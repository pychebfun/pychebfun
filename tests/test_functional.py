#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

# to prevent plots from popping up
import matplotlib
matplotlib.use('agg')

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


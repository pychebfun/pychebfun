#!/usr/bin/env python
# coding: UTF-8
from __future__ import division

import os
import unittest

import matplotlib
matplotlib.interactive(False)

class TestExamples(unittest.TestCase):
    pass


def _get_test(file_name):
    def test_run(self):
        """
        Check that the examples can be executed.
        """
        with open(file_name) as f:
            code = compile(f.read(), file_name, 'exec')
            exec(code, {})
    return test_run

here = os.path.dirname(__file__)
example_folder = os.path.join(here,os.path.pardir,'examples')
files = os.listdir(example_folder)
for example in files:
    test_name, ext = os.path.splitext(example)
    if ext == '.py':
        file_name = os.path.join(example_folder,example)
        setattr(TestExamples, test_name, _get_test(file_name))


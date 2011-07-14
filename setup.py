#!/usr/bin/env python
"""
Setup script for pychebfun

This uses Distutils, the standard Python mechanism for installing packages.
For the easiest installation just type::

    python setup.py install

(root privileges probably required). If you'd like to install in a custom
directory, such as your home directoy, type the following to install 
pychebfun under `<dir>/lib/python`::

    python setup.py install --home=<dir>

In addition, there are some other commands::

    python setup.py clean    -    clean all trash (*.pyc, emacs backups, etc.)
    python setup.py test     -    run test suite

"""

from distutils.core import setup
from distutils.core import Command

import sys
import pychebfun



# Check for Python version
if sys.version_info[:2] < (2,4):
    print "pychebfun require Python 2.4 or newer. Python %d.%d detected." %\
          sys.version_info[:2]
    sys.exit(-1)



# Define behavior of additional commands (clean, test, etc.)
class clean(Command):
    """
    Cleans the *pyc, emacs, and build results so you shold get the same copy
    as in the VCS. (Mercurial)
    """

    description  = 'remove all build and trash files'
    user_options = [("all", "a", "the same")]

    def initialize_options(self):
        self.all = None
        
    def finalize_options(self):
        pass

    def run(self):
        import os
        os.system("rm -fr ./*.pyc ./*~ ./*/*.pyc ./*/*~ ./*/*/*.pyc ./*/*/*~")
        os.system("rm -fr build")
        os.system("rm -fr dist")
        os.system("rm -fr doc/_build")

class test_pychebfun(Command):
    """
    Run all tests under the `pychebfun/` directory.
    """
    
    description  = 'run all tests and doctests'
    user_options = []         # distutils complains if this is not here

    def __init__(self, *args):
        self.args = args[0]   # so we can pass it to other classes
        Command.__init__(self, *args)

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        if pychebfun.test():
            # all regular tests ran successfully so run some doctests
            pychebfun.doctest()



# Define included modules and tests
modules = [
    'pychebfun.core',
    'pychebfun.analysis',
    'pychebfun.utilities',
    ]

tests = [
    'pychebfun.core.tests',
    'pychebfun.analysis.tests',
    'pychebfun.utilities.tests',
    ]

setup(
    name         = 'pychebfun',
    version      = pychebfun.__version__,
    description  = 'Python Chebyshev Functions',
    author       = 'Chris Swierczewski',
    author_email = 'cswiercz@gmail.com',
    license      = 'GPL v.3',
    url          = 'http://code.google.com/p/pychebfun',
    packages     = ['pychebfun'] + modules + tests,
    ext_modules  = [],
    cmdclass     = {'test':  test_pychebfun,
                    'clean': clean,
                    },
)



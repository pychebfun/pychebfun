#!/usr/bin/env python

from distutils.core import setup

import pychebfun


setup(
    name         = 'pychebfun',
    version      = pychebfun.__version__,
    description  = 'Python Chebyshev Functions',
    author = 'Olivier Verdier, Chris Swierczewski',
    url = 'https://github.com/olivierverdier/pychebfun',
    license      = 'GPL v.3',
    keywords = ['Math', 'Chebyshev', 'chebfun',],
    packages=['pychebfun',],
    classifiers = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering :: Mathematics',
    ],
    )

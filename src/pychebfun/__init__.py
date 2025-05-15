"""
pychebfun - Python Chebyshev Polynomials

See the webpage for more information and documentation:

    http://code.google.com/p/pychebfun
"""

__version__ = "0.3"


from .plotting import plot, compare
from .chebfun import Chebfun

__all__ = ['plot', 'compare', 'Chebfun']

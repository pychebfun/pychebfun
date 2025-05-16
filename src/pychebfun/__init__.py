"""
pychebfun - Python Chebyshev Polynomials

See the webpage for more information and documentation:

    http://code.google.com/p/pychebfun
"""

__version__ = "0.3"


from .chebfun import Chebfun
from .plotting import compare, plot

__all__ = ["Chebfun", "compare", "plot"]

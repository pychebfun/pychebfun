from typing import Callable, Optional
from .chebfun import Chebfun
from .polyfun import Polyfun
import numpy as np

# ----------------------------------------------------------------
# Constructor inspired by the Matlab version
# ----------------------------------------------------------------

def chebfun(f: Optional[Callable]=None, domain: tuple=(-1,1), N: Optional[int]=None, chebcoeff:  Optional[np.ndarray]=None,):
    """
    Create a Chebyshev polynomial approximation of the function $f$ on the interval :math:`[-1, 1]`.

    :param callable f: any univariate real numerical function
    :param int N: (default = None)  specify number of interpolating points
    :param np.array chebcoeff: (default = np.array(0)) specify the coefficients
    """

    # Chebyshev coefficients
    if chebcoeff is not None:
        return Chebfun.from_coeff(chebcoeff, domain)

    # another instance
    if isinstance(f, Polyfun):
        return Chebfun.from_fun(f)

    # callable
    if hasattr(f, '__call__'):
        return Chebfun.from_function(f, domain, N)

    # from here on, assume that f is None, or iterable
    if np.isscalar(f):
        f = [f]

    try:
        iter(f) # interpolation values provided
    except TypeError:
        pass
    else:
        return Chebfun(f, domain)

    raise TypeError('Impossible to initialise the object from an object of type {}'.format(type(f)))

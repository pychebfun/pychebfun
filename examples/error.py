from pychebfun import Chebfun, compare
import numpy as np

# Construct a Python function f and the vector of points at which we want 
# to plot it.
def f(x):
    return np.sin(6*x) + np.sin(30*np.exp(x))
x = np.linspace(-1,1,1000)

# Constuct the O(dx^-16) "spectrally accurate" chebfun p and compute the error
# between p and f at each point in the domain
p = Chebfun.from_function(f)
compare(p, f)

from scipy import *
from matplotlib.pyplot import figure, subplot, plot, show
from pychebfun import *

# Construct a Python function f and the vector of points at which we want 
# to plot it.
f = lambda x: sin(6*x) + sin(30*exp(x))
x = linspace(-1,1,1000)

# Constuct the O(dx^-16) "spectrally accurate" chebfun p and compute the error
# between p and f at each point in the domain
p   = chebfun(f)
compare(p, f)

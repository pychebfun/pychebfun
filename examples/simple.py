from pylab     import *
from pychebfun import *

# Construct a Python function f and the vector of points at which we want 
# to plot it.
f = lambda x: sin(6*x) + sin(30*exp(x))
x = linspace(-1,1,1000)

# Plot f on the above points
hold(True)
plot(x,f(x),'k',linewidth=10,alpha=0.3, label="Actual $f$")

# Construct a chebfun interpolation on 20, 40, and 60 points. Evaluate the 
# interpolations at the above vector of points and plot.
interps   = [20,40,60]
ps = [chebfun(f,N=N) for N in interps]
for p in ps:
    label = "Chebfun Interpolant: $N=%d$" % len(p)
    p.plot(linewidth=3, label=label)

legend()
show()


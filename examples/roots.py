from pylab import *
from pychebfun import *
from scipy.optimize import *

f = lambda x: sin(6*x) + sin(30*exp(x))
x = linspace(-1,1,5000)

# Computing the roots of the corresponding chebfun. Uses the specrally accurate
# Chebyshev expansion.
p = chebfun(f)
r = p.roots()

print "Roots (Chebfun):", r
print "f(r) =", f(r)
print "p(r) =", p(r)

plot(x,f(x), linewidth=3,label="$f(x)$")
plot(r,f(r),'k.', markersize=10, label="Roots of $f$")

legend()


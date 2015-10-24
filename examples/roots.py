import matplotlib.pyplot as plt
import numpy as np
from pychebfun import *
from scipy.optimize import *

f = lambda x: np.sin(6*x) + np.sin(30*np.exp(x))
x = np.linspace(-1,1,5000)

# Computing the roots of the corresponding chebfun. Uses the specrally accurate
# Chebyshev expansion.
p = chebfun(f)
r = p.roots()

print("Roots (Chebfun):", r)
print("f(r) =", f(r))
print("p(r) =", p(r))

plt.plot(x,f(x), linewidth=3,label="$f(x)$")
plt.plot(r,f(r),'k.', markersize=10, label="Roots of $f$")

plt.legend()


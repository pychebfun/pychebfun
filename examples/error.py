from pylab import *
from pychebfun import *

# Construct a Python function f and the vector of points at which we want 
# to plot it.
f = lambda x: sin(6*x) + sin(30*exp(x))
x = linspace(-1,1,1000)

# Constuct the O(dx^-16) "spectrally accurate" chebfun p and compute the error
# between p and f at each point in the domain
p   = Chebfun(f)
err = abs(f(x)-p(x))


# Plot the functions o top of each other and the abs error at each point
figure(1)
subplot(2,1,1)
plot(x,f(x),'k',linewidth=10,alpha=0.3)
plot(x,f(x),'r',linewidth=3)

subplot(2,1,2)
plot(x,err,'k')

show()

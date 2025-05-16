# pychebfun - Python Chebyshev Functions

[![pychebfun](https://github.com/pychebfun/pychebfun/actions/workflows/python_package.yml/badge.svg)](https://github.com/pychebfun/pychebfun/actions/workflows/python_package.yml)
[![codecov](https://codecov.io/gh/pychebfun/pychebfun/graph/badge.svg?token=NhTZBV0Q20)](https://codecov.io/gh/pychebfun/pychebfun)
![Python version](https://img.shields.io/badge/Python-3.9%20|%203.10%20|%203.11%20|%203.12-blue.svg?logo=python&logoColor=gold)

## About

To learn more about pychebfun, have a look at the [IPython notebook illustrating the original paper by Battles and Trefethen](https://gist.github.com/olivierverdier/50b359fbddfada213fa75e596a97b7c7#file-pychebfun_demo-ipynb).

## Getting Started

This is a minimal documentation about `pychebfun`.

As a minimal example, you can run the following:

```python
import numpy as np; from pychebfun import *
# define a function
f = Chebfun.from_function(lambda x:np.tan(x+1/4) + np.cos(10*x**2 + np.exp(np.exp(x))))
# evaluate at some point in [-1, 1]
f(.5)
f(np.linspace(-.5, .5, 200))
# plot it:
plot(f)
```

![Example](https://github.com/pychebfun/pychebfun/raw/main/images/ex1.png)

One can also use the general constructor `chebfun`:

```python
f = chebfun(lambda x:np.tan(x+1/4) + np.cos(10*x**2 + np.exp(np.exp(x))))
```

Note that one can could have defined the function `f` in a more intuitive manner by

```python
x = Chebfun.identity()
f = np.tan(x+1/4) + np.cos(10*x**2 + np.exp(np.exp(x)))
```

It is possible to multiply, add, subtract chebfuns between themselves and also with scalars:

```python
g = 2*np.sin(10*np.pi*x)
f+g
1+f
f-g
2*f*g - 1
```

One can find all the roots of a function with `roots`:

```python
f.roots() # all the roots of f on [-1, 1]
```

One can compute the integral of f:

```python
f.sum() # integral of f from -1 to 1
f.dot(g) # integral of f.g from -1 to 1
```

An arbitrary function can be differentiated and integrated:

```python
f.differentiate() # derivative of f
f.integrate() # primitive of f: ∫_0^x f(y) dy
```

You can see in [this example][5] how to compute the maxima and minima of an arbitrary function by computing the zeros of its derivative.
![Extrema](https://github.com/pychebfun/pychebfun/raw/main/images/extrema.png)

One can also have vector coefficients:

```python
def circle(x):
	return np.array([np.cos(np.pi*x), np.sin(np.pi*x)],).T
c = Chebfun.from_function(circle)
plot(c)
```

![Example](https://github.com/pychebfun/pychebfun/raw/main/images/circle.png)

If you are interested in experimenting with the innards of chebfun, you should be aware of the following functions:

```python
Chebfun.basis(10) # Chebyshev polynomial of degree 10
Chebfun.from_coeff([0.,1,2]) # Chebfun with prescribed chebcoeffs
Chebfun.interpolation_points(10) # 10 Chebyshev interpolation points in [-1, 1]
Chebfun.polyfit([1.,2]) # compute Chebyshev coefficients given values at Chebyshev points
Chebfun.polyval([1., 2.]) # compute values at Chebyshev points given Chebyshev coefficients
```

You should also take a look at the [examples][4] bundled with this project.
![Example](https://github.com/pychebfun/pychebfun/raw/main/images/example.png)

The pychebfun project is based on the mathematical work of Battles and Trefethen et. al. yet is optimized to take advantage of the tools in the Numpy/Scipy and Sage libraries. This project is solely for the educational purposes of the owner and is not meant to compete with the Matlab library created by Battles and Trefethen. Any questions regarding the Chebfun package for Matlab should be directed towards the [Chebfun team][2].

Pychebfun was started by [Chris Swierczewski][3] from the Applied Mathematics department at the University of Washington in Seattle, Washington, and is currently maintained by [Olivier Verdier][1].

This work is licensed under the GNU General Public
License v2.

## Installation

See 'INSTALL' located in this directory.

## Current Maintainer

- Olivier Verdier <olivier.verdier@gmail.com>

## Contributors

- Chris Swierczewski <cswiercz@gmail.com>

## Repository

pychebfun is hosted at http://github.com/pychebfun/pychebfun.

[1]: https://github.com/olivierverdier
[2]: https://www.chebfun.org/
[3]: https://github.com/cswiercz
[4]: https://github.com/pychebfun/pychebfun/tree/main/examples
[5]: https://github.com/pychebfun/pychebfun/tree/main/examples/extrema.py

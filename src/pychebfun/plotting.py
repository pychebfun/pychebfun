"""
Chrbfun plotting tools

CREATED BY:
----------

    -- Chris Swierczewski <cswiercz@gmail.com>

"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

from .chebfun import Chebfun


def chebpolyplot(f, Nmax=100, normpts=1000, ord=2, points_only=False):  # noqa: A002
    """
    Plots the number of Chebyshev points vs. norm accuracy of the
    chebfun interpolant.

    INPUTS:

        -- f: Python, Numpy, or Sage function

        -- Nmax: (default = 100) maximum number of interpolating points

        -- normpts: (default = 1000) number of sample points to take
                    when computing the norm

        -- ord: (default = None) {1,-1,2,-2,inf,-inf,'fro'} order of the norm

        -- compare: (default = False) compare normal with Chebyshev interpol

        -- points_only: (default = False) return only the plot points. do
                        not actually plot the graph

    OUTPUTS:

        -- Nvals: array of interpolating point numbers

        -- normvalscheb: array of norm values from Chebyshev interpolation

        -- normvalsequi: array of norm values from equidistant interpolation
    """

    x = np.linspace(-1, 1, normpts)
    Nvals = range(10, Nmax, 10)
    normvalscheb = [la.norm(f(x) - Chebfun.from_function(f, N=n)(x), ord=ord) for n in Nvals]

    # plot this
    if not points_only:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(Nvals, np.log10(normvalscheb), "r", label="Chebyshev Interpolation")
        ax.plot(Nvals, np.log10(normvalscheb), "r.", markersize=10)
        ax.set_xlabel("Number of Interpolating Points")
        ax.set_ylabel(f"{ord!s}-norm Error ($\\log_{{10}}$-scale)")
        ax.legend(loc="best")

    return ax


# ----------------------------------------------------------------
# Plotting Methods
# ----------------------------------------------------------------

plot_res = 1000


def dimension_info(poly):
    """
    Dimension information of the fun.
    """
    vals = poly.values
    # "local" degree of freedom; whether it is a complex or real fun
    t = vals.dtype.kind
    dof = 2 if t == "c" else 1
    # "global" degree of freedom: the dimension
    shape = np.shape(vals)
    dim = 1 if len(shape) == 1 else shape[1]
    return dim, dof


def plot_data(poly, resolution):
    """
    Plot data depending on the dimension of the fun.
    """
    ts = get_linspace(poly.domain, resolution)
    values = poly(ts)
    dim, dof = dimension_info(poly)
    if dim == 1 and dof == 1:  # 1D real
        xs = ts
        ys = values
        xi = poly._ui_to_ab(poly.p.xi)
        yi = poly.values
        d = 1
    elif dim == 2 and dof == 1:  # 2D real

        def xf(v):
            return v[:, 0]

        def yf(v):
            return v[:, 1]

        d = 2
    elif dim == 1 and dof == 2:  # 1D complex
        xf = np.real
        yf = np.imag
        d = 2
    else:
        raise ValueError("Too many dimensions to plot")  # noqa: TRY003
    if d == 2:
        xs, ys = xf(values), yf(values)
        xi, yi = xf(poly.values), yf(poly.values)
    return xs, ys, xi, yi, d


def plot(poly, ax=None, with_interpolation_points=False, *args, **kwargs):
    """
    Plot the fun with the additional arguments args, kwargs.
    """
    xs, ys, xi, yi, d = plot_data(poly, plot_res)
    axis = plt.gca() if ax is None else ax
    axis.plot(xs, ys, *args, **kwargs)
    if with_interpolation_points:
        current_color = axis.lines[-1].get_color()  # figure out current colour
        axis.plot(xi, yi, marker=".", linestyle="", color=current_color)
    plt.plot()
    if d == 2:
        axis.axis("equal")
    return axis


def chebcoeffplot(poly, *args, **kwds):
    """
    Plot the coefficients.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    coeffs = poly.coefficients()
    data = np.log10(np.abs(coeffs))
    ax.plot(data, "r", *args, **kwds)
    ax.plot(data, "r.", *args, **kwds)

    return ax


def compare(poly, f, *args, **kwds):
    """
    Plots the original function against its fun interpolant.

    INPUTS:

        -- f: Python, Numpy, or Sage function
    """
    a, b = poly.domain
    x = np.linspace(a, b, 10000)
    fig = plt.figure()
    ax = fig.add_subplot(211)

    ax.plot(x, f(x), "#dddddd", linewidth=10, label="Actual", *args, **kwds)  # noqa: B026
    label = f"Interpolant (d={poly.size()})"
    plot(poly, color="red", label=label, *args, **kwds)  # noqa: B026
    ax.legend(loc="best")

    ax = fig.add_subplot(212)
    ax.plot(x, abs(f(x) - poly(x)), "k")

    return ax


def get_linspace(domain, resolution):
    """
    Get a sample of points in the domain with the given resolution.
    """
    a, b = domain
    return np.linspace(a, b, resolution)

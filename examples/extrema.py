#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

from pychebfun import Chebfun, plot

f = Chebfun.from_function(lambda x: np.sin(6 * x) + np.sin(30 * np.exp(x)))

r = f.roots()
fd = f.differentiate()
fdd = fd.differentiate()
e = fd.roots()
ma = e[fdd(e) <= 0]
mi = e[fdd(e) > 0]


def plot_all():
    plot(f, with_interpolation_points=False)
    plt.plot(r, np.zeros_like(r), linestyle="", marker="o", color="white", markersize=8)
    plt.plot(ma, f(ma), linestyle="", marker="o", color="green", markersize=8)
    plt.plot(mi, f(mi), linestyle="", marker="o", color="red", markersize=8)
    plt.grid()


plot_all()
plt.title("Maxima, minima and zeros of an arbitrary function")

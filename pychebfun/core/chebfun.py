"""
Chebfun class

CREATED BY:
-----------

    -- Chris Swierczewski <cswiercz@gmail.com>

"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import sys

from scipy.interpolate import BarycentricInterpolator as Bary
from scipy.fftpack     import fft            # implement DCT routine later

emach     = sys.float_info.epsilon                        # machine epsilon

class Chebfun(object):
    """
    Construct a Lagrange interpolating polynomial over the Chebyshev points.

    """
    max_nb_dichotomy = 12 # maximum number of dichotomy of the interval

    def __init__(self, f, N=0, spacing='chebyshev', verbose=False, ai=None):
        """
        Create a Chebyshev polynomial approximation of the function $f$ on the
        interval $[a,b]$.

    
        INPUTS:

            -- f: Python, Numpy, or Sage function

            -- N: (default = None)  specify number of interpolating points
            
            -- spacing: (default = 'chebyshev') interpolation point spacing

            -- verbose: (default = False) print convergence information

        EXAMPLES:
        """
        self.fun     = f
        self.spacing = spacing
        self.verbose = verbose
        
        #
        # If user provides a list of Chebyshev expansion coefficients
        # use them to generate a chebfun
        #
        if ai != None:
            self.ai = ai
            self.N  = int((len(ai)+1)/2)
            self.x  = [np.cos(j*np.pi/self.N) for j in range(self.N+1)]
            self.f  = f(self.x)
            self.p  = Bary(self.x, self.f)
        #
        # If user inputs a specific number for N then simply create a
        # chebfun with that many interpolating points. This is mostly
        # used for accuracy analysis.
        #
        elif N:
            self.N = N
            if spacing == 'chebyshev': 
                self.x = self.chebyshev_points(N)
            else:
                self.x = np.linspace(-1,1,self.N)

            self.f = f(self.x)
            self.p  = Bary(self.x, self.f)

            self.ai = self.fft_data(f, N)
            return None

        else:
            #
            # Otherwise, construct machine precision chebfun interpolant
            # (Primary Algorithm)
            #
            # (2) Loop until convergence condition
            #
            for k in xrange(2,self.max_nb_dichotomy):
                N = pow(2,k)
                
                # 2) Perform FFT and obtain Chebyshev Coefficients
                #    NOTE: We should write a fast cosine transform
                #          routine instead. This is a factor of two
                #          slower.
                fftdata = self.fft_data(f,N)


                # 3) Check for negligible coefficients
                #    If within bound: get negligible coeffs and bread
                #    Else:            loop
                bnd = 128*emach*abs(np.max(fftdata))
                if verbose:
                    print "\n===== STEP ====="
                    print "_______      N =", N
                    print "_______     ai =", fftdata
                    print "_______    bnd =", bnd
                    
                if np.all(abs(fftdata[-2:]) < bnd):
                    break
            else:
                raise Exception('No convergence')


            # End of convergence loop: construct polynomial
            [inds]  = np.where(abs(fftdata) > bnd)
            self.N = inds[-1]
            self.ai = fftdata[:(self.N+1)]
            self.x  = self.chebyshev_points(self.N)
            self.f  = f(self.x)
            self.p  = Bary(self.x, self.f)
            
            if verbose:
                print
                print "========================="
                print "       CONVERGENCE       "
                print "========================="
                print "______     bnd =", bnd
                print "______      ai =", self.ai
                print "______       N =", self.N
                print


    def even_data(self, data):
        """
        Construct Extended Data Vector (equivalent to creating an
        even extension of the original function)
        """
        return np.hstack([data,data[-2:0:-1]])

    def chebyshev_points(self, N):
        """
        N+1 Chebyshev points in [-1,1], boundaries included
        """
        return np.cos(np.arange(N+1)*np.pi/N)

    def sample(self, f, N):
        x = self.chebyshev_points(N)
        return f(x)

    def fft(self, data):
        N = len(data)//2
        fftdata     = np.real(fft(data)[:N+1])
        fftdata     /= N
        fftdata[0]  /= 2.
        fftdata[-1] /= 2.
        return fftdata

    def fft_data(self, f, N):
        sampled = self.sample(f,N)
        evened = self.even_data(sampled)
        return self.fft(evened)

    def __repr__(self):
        return "<Chebfun({0})>".format(self.N)


    #
    # Basic Operator Overloads
    #
    def __call__(self, x):
        return self.p(x)

    def __len__(self):
        return self.p.n

    def __add__(self, other):
        """
        Chebfun addition.

        Add the underlying functions.

        EXAMPLES::
        
            >>> 1+1
            2
        """
        if not isinstance(other,Chebfun):
            other = Chebfun(lambda x: other(x),
                            N = self.N,
                            verbose=self.verbose)

        return Chebfun(lambda x: self.fun(x) + other.fun(x),
                       N = self.N,
                       verbose=self.verbose)


    def __sub__(self, other):
        """
        Chebfun subtraction.
        """
        if not isinstance(other,Chebfun):
            other = Chebfun(lambda x: other(x),
                            N = self.N,
                            verbose=self.verbose)

        return Chebfun(lambda x: self.fun(x) - other.fun(x),
                       N = self.N,
                       verbose=self.verbose)


    def __mul__(self, other):
        """
        Chebfun multiplication.
        """
        if not isinstance(other,Chebfun):
            other = Chebfun(lambda x: other(x),
                            N = self.N,
                            verbose=self.verbose)

        return Chebfun(lambda x: self.fun(x) * other.fun(x),
                       verbose=self.verbose)

    def __div__(self, other):
        """
        Chebfun multiplication.
        """
        if not isinstance(other,Chebfun):
            other = Chebfun(lambda x: other(x),
                            N = self.N,
                            verbose=self.verbose)

        return Chebfun(lambda x: self.fun(x) / other.fun(x),
                       N = self.N,
                       verbose=self.verbose)


    def __neg__(self):
        """
        Chebfun negation.
        """
        return Chebfun(lambda x: -self.fun(x),
                       N = self.N,
                       verbose=self.verbose)


    def sqrt(self):
        """
        Square root of Chebfun.
        """
        return Chebfun(lambda x: np.sqrt(self.fun(x)),
                       N = self.N,
                       verbose=self.verbose)

    def __abs__(self):
        """
        Absolute value of Chebfun. (Python)

        (Coerces to NumPy absolute value.)
        """
        return Chebfun(lambda x: np.abs(self.fun(x)),
                       N = self.N,
                       verbose=self.verbose)

    def abs(self):
        """
        Absolute value of Chebfun. (NumPy)
        """
        return self.__abs__()

    def sin(self):
        """
        Sine of Chebfun
        """
        return Chebfun(lambda x: np.sin(self.fun(x)),
                       N = self.N,
                       verbose=self.verbose)


    #
    # Numpy / Scipy Operator Overloads
    #

    def chebyshev_coefficients(self):
        return self.ai

    def integral(self):
        """
        Evaluate the integral of the Chebfun over the given interval using
        Clenshaw-Curtis quadrature.
        """
        val = 0
        for n in np.arange(self.N, step=2):
            ai = self.ai[n]
            if abs(ai) > emach:
                val += 2.0*ai/(1.0-n**2)

        return val


    def integrate(self):
        """
        Return the Chebfun representing the integral of self over the domain.

        (Simply numerically integrates the underlying Barcentric polynomial.)
        """
        return Chebfun(self.p.integrate)

 
    def derivative(self):
        return self.differentiate()

    def differentiate(self):
        """
        Return the Chebfun representing the derivative of self. Uses spectral
        methods for accurately constructing the derivative.
        """
        # Compute new ai by doing a backsolve

        # If a_i and b_i are the kth Chebyshev polynomial expansion coefficient
        # Then b_{i-1} = b_{i+1} + 2ia_i; b_N = b_{N+1} = 0; b_0 = b_2/2 + a_1

        # DOESNT WORK YET :( POSSIBLY DUE TO __init__?
        bi = np.array([0])
        for i in np.arange(self.N,1,-1):
            bi = np.append(bi,bi[0] + 2*self.ai[i])
        bi = np.append(bi,bi[0]/2 + self.ai[i])

        return Chebfun(self.fun, ai=bi)

    def roots(self):
        """
        Return the roots of the chebfun.
        """
        N            = len(self.ai)
        coeffs       = np.append(np.array([self.ai[N-k-1] for k in np.arange(N)]), self.ai[1:])
        coeffs[N-1] *= 2
        zNq          = np.poly1d(coeffs)
        return np.unique(np.array([np.real(r) for r in zNq.roots if abs(r) > 0.99999999 and abs(r) < 1.00000001]))
        
    plot_res = 1000

    def plot(self, *args, **kwargs):
        xs = np.linspace(-1,1,self.plot_res)
        plt.plot(xs,self(xs), *args, **kwargs)

    def chebcoeffplot(self, *args, **kwds):
        """
        Plot the coefficients.
        """
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        
        data = np.log10(np.abs(self.ai))
        ax.plot(data, 'r' , *args, **kwds)
        ax.plot(data, 'r.', *args, **kwds)

        return ax

    def compare(self,f,*args,**kwds):
        """
        Plots the original function against its chebfun interpolant.
        
        INPUTS:

            -- f: Python, Numpy, or Sage function
        """
        x   = np.linspace(-1,1,10000)
        fig = plt.figure()
        ax  = fig.add_subplot(211)
        
        ax.plot(x,f(x),'#dddddd',linewidth=10,label='Actual', *args, **kwds)
        ax.plot(x,self(x),'r', label='Chebfun Interpolant (N=%d)' %self.N, *args, **kwds)
        ax.plot(self.x,self.f, 'r.', *args, **kwds)
        ax.legend(loc='best')

        ax  = fig.add_subplot(212)
        ax.plot(x,abs(f(x)-self(x)),'k')

        return ax




#!/usr/bin/env python
"""
:mod:`puwr` -- Error analysis for Monte Carlo data in python
=================================================================
.. module:: puwr
   :platform: Unix, Windows
   :synopsis: MC error analysis as described by Wolff in CPC **156**,
     143 (2004)

This module implements the method to estimate the autocorrelation time
as described in [1]_.
"""

import numpy as np
from scipy import signal
try:
    import matplotlib.pyplot as plt
except:
    pass

def means(data):
    r"""Calculate per-observable means.
    
    :param data: The input data is assumed to be in the format

      .. math:

        data[\alpha][r][i] =  a_\alpha^{i,r},

      where :math:`i` labels the measurements, :math:`r` the replica
      and :math:`\alpha` the observables.
    """
    return np.array([np.mean(np.hstack(i)) for i in data])

class DataSanityCheckFail:
    def __init__(self, s):
        self.value = s
    def __str__(self):
        return repr(self.value)

class DataInfo(object):
    def __init__(self, data):
        self.nobs = len(data) # number of observables
        self.R = len(data[0]) # number of replica
        # number of measurements per replicum
        self.Nr = [len(i) for i in data[0]]
        # check sanity of data
        for d in data:
            if [len(i) for i in d] != self.Nr:
                raise DataSanityCheckFail(
                    "Inconsistent number of measurements/replicum")
        self.N = sum(self.Nr)

def deriv(f, alpha, h):
    r"""Calculates the partial numerical derivative of a function.

    :param f: Function.
    :param alpha: Calculate partial derivative ith
      respect to the alpha-th argument.
    :param h: Step-size.
    :returns: The derivative (function).
    """
    def df(*a):
        args = list(a)
        args[alpha] += h
        result = f(*args)
        args[alpha] -= 2*h
        return (result - f(*args))/2./h
    return df


class DataProject:
    r"""Class to calculate the projected data

    ..math::

      a_f^{i,r} = \sum_\alpha \bar \bar f_\alpha a_\alpha^{i,r}

    for arbitrary functions.
    :param data: The input data is assumed to be in the format

      .. math:

        data[\alpha][r][i] =  a_\alpha^{i,r},

      where :math:`i` labels the measurements, :math:`r` the replica
      and :math:`\alpha` the observables.
    """
    
    def __init__(self, data):
        self.d = DataInfo(data)
        self.data = data
        # compute the step size h = sqrt( Gamma_aa / N )
        # first Gamma_aa
        self.m = means(data)
        G = np.array([np.sum ( np.sum( (rep - omean)**2 ) 
                           for rep in obs )
                      for obs, omean in zip(data, self.m)]) / self.d.N
        self.h = np.sqrt(G/self.d.N)
    def project(self, f):
        """Calculate the actual projected data w.r.t. the function
        :math:`f(A_1, ..., A_n)`,

        .. math::
        
          a_f^{i,r} = \sum_\alpha \bar \bar f_\alpha a_\alpha^{i,r},\\
          f_\alpha = d f / d A_\alpha

        :param f: Function.
        :returns: Projected data :math:`a_f^{i,r}` (array).
        """
        if isinstance(f, int):
            return self.data[f]
        # calculate the derivatives
        df = [deriv(f, alpha, h) for alpha, h in enumerate(self.h)]
        fa = [dfi(*self.m) for dfi in df]
        # calculate a_f
        return np.sum( np.array( [rep * falpha for rep in obs] )
                          for obs, falpha in zip (self.data, fa) )

def gamma(data, f):
    # calculate the projected data
    d = DataProject(data)
    af = d.project(f)
    # calculate the mean of the projected data
    # note we now have only one observable, a_f
    om = means([af,])[0]
    Gtil = np.sum( signal.fftconvolve(rep - om, 
                                      rep[::-1] - om)[len(rep)-1::-1]
                   for rep in af)
    return Gtil * np.array([1./(d.d.N - d.d.R*t) 
                            for t in range(len(Gtil))]), d.d, om

def tauint(data, f, full_output = False, plots=False):
    G, d, means = gamma(data, f)
    s = 0
    sums = [0,]
    for W in range(1, len(G)):
        s += G[W]
        sums.append(s)
    tint = [(.5*G[0] + i)/G[0] for i in sums]
    # suppress errors from overflow
    np.seterr(over = 'ignore')
    g = np.exp(-np.arange(1,len(tint))/tint[1:]) - \
        tint[1:]/np.sqrt(np.arange(1,len(tint))*d.N)
    np.seterr(over = 'warn')
    W = np.where(g < np.zeros(len(g)))[0][0]
    tint = np.array(tint)
    dtint = tint * 2 * \
        np.sqrt((np.arange(len(tint)) -  tint + .5)/d.N)
    # make a plot
    try:
        if plots:
            xmax = int(W*1.3)
            step = int(np.ceil(W/20)) or 1
            fig = plt.figure(1)
            tplt = fig.add_subplot(211)
            tplt.set_ylabel(r'$\tau_{\mathrm{int}}$')
            tplt.set_xlabel(r'$W$')
            plt.errorbar(range(xmax)[::step], tint[:xmax:step], 
                         dtint[:xmax:step], fmt="o", color='b')
            plt.axvline(W, color='r')
            Gplt = fig.add_subplot(212)
            Gplt.set_ylabel(r'$\Gamma$')
            Gplt.set_xlabel('$W$')
            plt.errorbar(range(xmax)[::step], G[:xmax:step], 
                         fmt="o", color='b')
            plt.axvline(W+1, color='r')
            plt.show()
    except NameError: # no matplotlib
        pass

    if not full_output:
        return means, \
        np.sqrt(G[0]/d.N*2*(tint[W+1])), tint[W+1], dtint[W+1]
    else:
        return means, \
        np.sqrt(G[0]/d.N*2*(tint[W+1])), tint[W+1], dtint[W+1],\
        G, W+1 


def correlated_data(tau = 5, n = 10000):
    r"""Generate correlated data as explained in the appendix of
    [1]_. One draws a sequence of :math:`n` normally distributed
    random numbers :math:`\eta_i, i = 1,\ldtos,n` with unit variance
    and vanishing mean. From this one constructs

    .. math::

      \nu_1 = \eta_1,\quad \nu_{i+1} = \sqrt{1 - a^2} \eta_{i+1} + a
      \nu_i\,,\\
      a = \frac {2 \tau - 1}{2\tau + 1}, \quad \tau \geq \frac 1 2\,,

    where :math:`\tau` is the autocorrelation time.

    :param tau: Target autocorrelation time.
    :param n: Number of data points to generate.

    .. [1] U. Wolff [ALPHA Collaboration], *Monte Carlo errors with
           less errors*, Comput. Phys. Commun.  **156**, 143 (2004)
           ``[hep-lat/0306017]``.
    """
    eta = np.random.rand(n)
    a = (2. * tau - 1)/(2. * tau + 1)
    asq = a**2
    nu = np.zeros(n)
    nu[0] = eta[0]
    for i in range(1, n):
        nu[i] = np.sqrt(1 - asq)*eta[i] + a * nu[i-1]
    return [[nu*0.2 + 1]]

def idf(n):
    return lambda *a : a[n]

if __name__ == "__main__":
    mean, err, tint, dtint, G, W = tauint(correlated_data(), 0, True)
    print " mean =", mean
    print "error =", err
    print " tint = {0} +/- {1}".format(tint, dtint)

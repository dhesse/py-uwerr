#!/usr/bin/env python
"""
:mod:`puwr` -- Error analysis for Monte Carlo data in python
=================================================================
.. module:: puwr
   :platform: Unix, Windows
   :synopsis: MC error analysis as described by Wolff in CPC **156**,
     143 (2004)

This module implements the method to estimate the autocorrelation time
as described in [1]_. In the reference, the author provides an
implementation in MATLAB. Our aim is to provide an alternative that
does not depend on proprietary software. Make sure you read [1]_
carefully to know what exactly this package does and how to interpret
its output. Note also that there is a similar method described in [2]_
which may be more suitable in your case.

.. [1] U. Wolff [**ALPHA** Collaboration], *Monte Carlo errors with less
   errors*, Comput. Phys. Commun.  **156**, 143 (2004)
   ``[hep-lat/0306017]``.

.. [2] S. Schaefer *et al.*  [**ALPHA** Collaboration],
  *Critical slowing down and error analysis in lattice QCD simulations,*
  Nucl. Phys. B **845**, 93 (2011) ``[arXiv:1009.5228 [hep-lat]]``.

"""

import numpy as np
from scipy import signal
try:
    import matplotlib.pyplot as plt
except:
    pass

__version__ = "0.1"


def means(data):
    r"""Calculate per-observable means::

      >>> from puwr import means
      >>> import numpy as np
      >>> data = [[np.linspace(0,10,20)],[np.linspace(10,20,20)]]
      >>> means(data)
      array([ 5., 15.])
    
    :param data: The input data is assumed to be in the format
      :math:`\mathtt{data[}\alpha\mathtt{][r][i]} = a_\alpha^{i,r}`
      where :math:`i` labels the measurements, :math:`r` the replica
      and :math:`\alpha` the observables.
    """
    return np.array([np.mean(np.hstack(i)) for i in data])

class DataSanityCheckFail(Exception):
    """Exception class."""
    def __init__(self, s):
        self.value = s
    def __str__(self):
        return repr(self.value)

class DataInfo(object):
    """Given an array of data points, extract information like the
    number of observables, replica, and measurements per replicum. If
    there is not the same number of measurements per replicum for each
    observable, an exception is raised::

      >>> from puwr import DataInfo, correlated_data
      >>> d = DataInfo([[correlated_data(5,20)[0][0],
      ...                correlated_data(6,20)[0][0]]])
      >>> d.nobs  # 1 observable
      1
      >>> d.R     # 2 'replica'
      2
      >>> d.Nr    # 20 measurements / replicum
      [20, 20]
      >>> DataInfo([correlated_data(5,20)[0],
      ...           correlated_data(6,29)[0]])
      Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
        File "puwr.py", line 61, in __init__
          self.R = len(data[0]) # number of replica
      puwr.DataSanityCheckFail: 'Inconsistent number of measurements/replicum'

    :param data: The input data is assumed to be in the format
      :math:`\mathtt{data[}\alpha\mathtt{][r][i]} = a_\alpha^{i,r}`
      where :math:`i` labels the measurements, :math:`r` the replica
      and :math:`\alpha` the observables.
    :raises: :class:`DataSanityCheckFail`
    """
    def __init__(self, data):
        self.nobs = len(data) # number of observables
        self.R = len(data[0]) # number of replica
        # number of measurements per replicum
        self.Nr = [len(i) for i in data[0]]
        # check sanity of data
        for d in data:
            if [len(i) for i in d] != self.Nr:
                msg = "Inconsistent number of measurements/replicum"
                raise DataSanityCheckFail(msg)
        self.N = sum(self.Nr)

def deriv(f, alpha, h):
    r"""Calculates the partial numerical derivative of a function.

    .. math::

      \partial_{i,h} f(x_0, \ldots, x_n) = \frac 1 {2h} \{ f(x_0,
      \dots, x_i + h, \ldots, x_n) - f(x_0,
      \dots, x_i - h, \ldots, x_n)\}.

    :param f: Function.
    :param alpha: Calculate partial derivative with
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

    .. math::

      a_f^{i,r} = \sum_\alpha \bar{\bar{f}}_\alpha a_\alpha^{i,r}\,,
      \quad f_\alpha = \frac{\mathrm d f}{\mathrm d A_\alpha}

    for arbitrary functions.

    :param data: The input data is assumed to be in the format
      :math:`\mathtt{data[}\alpha\mathtt{][r][i]} = a_\alpha^{i,r}`
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
        r"""Calculate the actual projected data w.r.t. the function
        :math:`f(A_1, ..., A_n)`,

        .. math::
        
          a_f^{i,r} = \sum_\alpha \bar{\bar{f}}_\alpha a_\alpha^{i,r}\,,
          \quad f_\alpha = \frac{\mathrm d f}{\mathrm d A_\alpha}

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
    r"""Calculates an estimator for the autocorrelation function

    .. math::

      \bar{\bar{\Gamma}}_F(t) =& \sum_{\alpha\beta}
        \bar{\bar{f}}_\alpha\,\bar{\bar{f}}_\beta
        \bar{\bar{\Gamma}}_{\alpha\beta}(t)\,,\\
      \bar{\bar{\Gamma}}_{\alpha\beta}(t) =&
        \frac 1 {N -R t}
        \sum_{r = 1}^R \sum_{i = 1}^{N_r -t}
        (a_\alpha^{i,r} - \bar{\bar{a}}_\alpha)
        (a_\beta^{i+t,r} - \bar{\bar{a}}_\beta)\,,\\
      f_\alpha = \frac{\mathrm d f}{\mathrm d A_\alpha}

    where :math:`f` is a function, :math:`i` labels the measurements,
    :math:`r` the replica, and :math:`\alpha` the observables.

    :param data: The input data is assumed to be in the format
      :math:`\mathtt{data[}\alpha\mathtt{][r][i]} = a_\alpha^{i,r}`
      where :math:`i` labels the measurements, :math:`r` the replica
      and :math:`\alpha` the observables.
    :param f: Function :math:`f` as above.
    """
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
    r"""Estimate the autocorrelation time of data as presented in
    [1]_.

    :param data: The input data is assumed to be in the format
      :math:`\mathtt{data[}\alpha\mathtt{][r][i]} = a_\alpha^{i,r}`
      where :math:`i` labels the measurements, :math:`r` the replica
      and :math:`\alpha` the observables.
    :param f: Function or integer. If a function is passed, the
      autocorrelation time for the secondary observable, defined by
      the function applied to the input data (with the primary
      observables in the order they are given in ``data`` passed as
      arguments to ``f``) is calculated. If an integer s passed, the
      auto-correlation time of ``data[f]`` is estimated.
    :param full_output: If set to ``True``, the autocorrelation matrix
      :math:`\bar{\bar{\Gamma}}` and the optimal window size :math:`W`
      will be appended to the output.
    :param plots: If set to ``True``, and if the ``matplotlib``
      package is installed, plots are produced that depict the
      autocorrelation matrix and estimated autocorrelation time
      vs. the window size.
    :returns: A tuple containing the mean, variance, estimated
      autocorrelation and the estimated error thereof.
    """
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
            plt.errorbar(list(range(xmax))[::step], tint[:xmax:step], 
                         dtint[:xmax:step], fmt="o", color='b')
            plt.axvline(W, color='r')
            Gplt = fig.add_subplot(212)
            Gplt.set_ylabel(r'$\Gamma$')
            Gplt.set_xlabel('$W$')
            plt.errorbar(list(range(xmax))[::step], G[:xmax:step], 
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
    random numbers :math:`\eta_i, i = 1,\ldots,n` with unit variance
    and vanishing mean. From this one constructs

    .. math::

      \nu_1 = \eta_1,\quad \nu_{i+1} = \sqrt{1 - a^2} \eta_{i+1} + a
      \nu_i\,,\\
      a = \frac {2 \tau - 1}{2\tau + 1}, \quad \tau \geq \frac 1 2\,,

    where :math:`\tau` is the autocorrelation time::

      >>> from puwr import correlated_data
      >>> correlated_data(2, 10)
      [[array([1.1703499 , 1.18119393, 1.17114224, 1.13142256, 1.09497294,
             1.18182216, 1.23490896, 1.28032049, 1.21973591, 1.15085657])]]

    :param tau: Target autocorrelation time.
    :param n: Number of data points to generate.
    """
    rng = np.random.default_rng(125)
    eta = rng.random(n)
    a = (2. * tau - 1)/(2. * tau + 1)
    asq = a**2
    nu = np.zeros(n)
    nu[0] = eta[0]
    for i in range(1, n):
        nu[i] = np.sqrt(1 - asq)*eta[i] + a * nu[i-1]
    return [[nu*0.2 + 1]]

def idf(n):
    """Project on n-th argument::
    
       idf(n) = lambda *a : a[n]
       
    :param n: Number of element to project on.
    """
    return lambda *a : a[n]

if __name__ == "__main__":
    mean, err, tint, dtint, G, W = tauint(correlated_data(), 0, True)
    print(" mean =", mean)
    print("error =", err)
    print(" tint = {0} +/- {1}".format(tint, dtint))

============================
Analysis of Monte Carlo data
============================

Author: Dirk Hesse <herr.dirk.hesse@gmail.com>

We implement the method to estimate autocorrelation times of Monte
Carlo data presented in 

U. Wolff [ALPHA Collaboration], *Monte Carlo errors with less errors*,
Comput. Phys. Commun.  **156**, 143 (2004) ``[hep-lat/0306017]``.

**PUBLICATIONS MAKING USE OF THIS CODE MUST CITE THE PAPER.**

The main objective is the following: Data coming from a Monte Carlo
simulation usually suffers from autocorrelation. It is not
straight-forward to estimate this autocorrelation, which is required
to give robust estimates for errors. This program implements a method
proposed by Wolff to estimate autocorrelations in a safe way.


Quick start
===========

Installation
------------

Just::

  $ pip install py-uwerr

should be enough to install this library from the PyPI.


Usage
-----

This package contains code to generate correlated data, so we can
conveniently demonstrate the basic functionality of the code in a
short example::

  >>> from puwr import tauint, correlated_data
  >>> correlated_data(2, 10)
  [[array([ 1.02833043,  1.08615234,  1.16421776,  1.15975754,
            1.23046603,  1.13941114,  1.1485227 ,  1.13464388,
            1.12461557,  1.15413354])]]
  >>> mean, delta, tint, d_tint = tauint(correlated_data(10, 200), 0)
  >>> print "mean = {0} +/- {1}".format(mean, delta)
  mean = 1.42726267057 +/- 0.03013853
  >>> print "tau_int = {0} +/- {1}".format(tint, d_tint)
  tau_int = 9.89344869217 +/- 4.10466090332

The data is expected to be in the format
``data[observable][replicum][measurement]``. See the documentation
that comes with this code for more information.


License
=======

See LICENSE file.

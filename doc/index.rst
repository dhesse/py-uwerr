.. py-uwerr documentation master file, created by
   sphinx-quickstart on Sat Nov 24 19:07:07 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to py-uwerr's documentation!
====================================

We implement the method to estimate autocorrelation times of Monte
Carlo data presented in [1]_ .

**PUBLICATIONS MAKING USE OF THIS CODE MUST CITE THE PAPER.**

Data generated with most common Monte Carlo methods used in lattice
field theory usually suffers from strong autocorrelation. It is not
straight-forward to estimate this autocorrelation, which is required
to give robust estimates for errors. This program implements a method
proposed by Wolff to estimate autocorrelations in a safe way.

Quick start
===================

The code can be grabbed directly from `github
<https://github.com/dhesse/py-uwerr>`_, either via ``ssh`` or http:

.. code-block:: shell

  git clone git@github.com:dhesse/py-uwerr.git  # OR
  git clone https://github.com/dhesse/py-uwerr.git

or downloaded as a `.zip archive
<https://github.com/dhesse/py-uwerr/archive/master.zip>`_.  The
package contains code to generate correlated data, so we can
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
below for more information.

Self tests
-------------------------------------

Note that there are unit-tests included in the code archive (located
in the sub-directory ``test``). They might serve as a first guide-line
how the code can be used.

.. automodule:: puwr
   :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


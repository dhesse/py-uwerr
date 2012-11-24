"""
:mod:`test_util` -- Tests for various utility functions.
===============================================================
.. module: test_util

This module contains test for some utility functions.
"""

from unittest import TestCase, main
import sys, os
sys.path.append(os.path.abspath(".."))
from puwr import means, deriv
import numpy as np
from math import sin, cos

class TestMeansFunc(TestCase):
    """Test the :class:`means` function."""
    def test_singe_bin_known_values(self):
        """Test with data with known mean values with one bin per
        observable."""
        data = [[np.linspace(0,10,20)],[np.linspace(10,20,20)]]
        mdata = means(data)
        self.assertAlmostEqual(mdata[0], 5.)
        self.assertAlmostEqual(mdata[1], 15.)
    def test_multiple_bins_known_values(self):
        """Test with data with known mean values with two bins."""
        data = [[np.linspace(0,10,20), np.linspace(10,20,20)]]
        mdata = means(data)
        self.assertAlmostEqual(mdata[0], 10.)



class TestDerivFunc(TestCase):
    """Test the numerical derivative (:class:`puwr.deriv`)."""
    def test_mononom(self):
        """Test with a constant, linear, quadratic and cubic
        function."""
        f = lambda w,x,y,z : x + y**2 + z**3
        dfw = deriv(f, 0, .01) # 0
        self.assertAlmostEqual(dfw(1,1,1,1), 0.)
        self.assertAlmostEqual(dfw(1,2,1,1), 0.)
        self.assertAlmostEqual(dfw(1,2,2,1), 0.)
        self.assertAlmostEqual(dfw(1,2,1,2), 0.)
        dfx = deriv(f, 1, .01) # 1 
        self.assertAlmostEqual(dfx(1,1,1,1), 1.)
        self.assertAlmostEqual(dfx(1,2,1,1), 1.)
        self.assertAlmostEqual(dfx(1,2,1,2), 1.)
        dfy = deriv(f, 2, .01) # 2*y
        self.assertAlmostEqual(dfy(1,1,1,1), 2.)
        self.assertAlmostEqual(dfy(1,1,2,1), 4.)
        dfz = deriv(f, 3, .0001) # 3 * z**2
        self.assertAlmostEqual(dfz(1,1,1,1), 3.)
        self.assertAlmostEqual(dfz(1,1,1,2), 12.)
    def test_sin(self):
        """Test with the sine function."""
        f = lambda x : sin(x)
        df = deriv(f, 0, .00001)
        for x in np.linspace(0,2,20):
            self.assertAlmostEqual(df(x), cos(x))
        
if __name__ == "__main__":
    main(verbosity = 2)

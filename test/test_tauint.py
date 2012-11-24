"""
:mod:`test_tauin` -- Tests for the main routine.
=============================================================
.. module: test_util

This module contains test for the main code. 
"""

from unittest import TestCase, main
import sys, os
sys.path.append(os.path.abspath(".."))
from puwr import tauint, correlated_data
import numpy as np

class TestPrimary(TestCase):
    """Tests if the tauint method returns the correct values."""
    def test_with_correlated_data_single_replicum(self):
        r"""Check the :class:`puwr.tauint` method with the
        :class:`puwr.correlated_data` method. Note that this is a
        statistical test with random numbers. We test for a
        :math:`3\sigma` deviation here, which is unlikely but not
        impossible. If this test fails, re-run a few times before
        making any conclusions."""
        fails = 0
        for known_tau in np.linspace(2,20,20):
            data = correlated_data(known_tau)
            mean, err, tau, dtau = tauint(data, 0)
            if tau - 3*dtau > known_tau or \
                tau + 3*dtau < known_tau:
                print tau - known_tau, dtau
                fails += 1
        self.assertGreaterEqual(1, fails)
    def test_with_correlated_data_multiple_replica(self):
        r"""Check the :class:`puwr.tauint` method with the
        :class:`puwr.correlated_data` method. Note that this is a
        statistical test with random numbers. We test for a
        :math:`3\sigma` deviation here, which is unlikely but not
        impossible. If this test fails, re-run a few times before
        making any conclusions."""
        fails = 0
        for known_tau in np.linspace(2,20,10):
            data = [[correlated_data(known_tau)[0][0],
                     correlated_data(known_tau)[0][0],
                     correlated_data(known_tau)[0][0]]]
            mean, err, tau, dtau = tauint(data, 0)
            if tau - 3*dtau > known_tau or \
                tau + 3*dtau < known_tau:
                print tau - known_tau, dtau
                fails += 1
        self.assertGreaterEqual(0, fails)
    

class SecondaryTest(TestCase):
    """Test the analysis of a secondary observable."""
    def test_secondary_multiple_replica_sum(self):
        """Test using the sum of two observables."""
        fails = 0
        for known_tau in np.linspace(2,20,10):
            data = [[correlated_data(known_tau)[0][0],
                     correlated_data(known_tau)[0][0],
                     correlated_data(known_tau)[0][0]],
                    [correlated_data(known_tau)[0][0],
                     correlated_data(known_tau)[0][0],
                     correlated_data(known_tau)[0][0]]]
            mean, err, tau, dtau = tauint(data, lambda x, y: x + y)
            if tau - 3*dtau > known_tau or \
                tau + 3*dtau < known_tau:
                print tau - known_tau, dtau
                fails += 1
        self.assertGreaterEqual(0, fails)
    def test_secondary_multiple_replica_product(self):
        """Test using the product of two observables."""
        fails = 0
        for known_tau in np.linspace(2,20,10):
            data = [[correlated_data(known_tau)[0][0],
                     correlated_data(known_tau)[0][0],
                     correlated_data(known_tau)[0][0]],
                    [correlated_data(known_tau)[0][0],
                     correlated_data(known_tau)[0][0],
                     correlated_data(known_tau)[0][0]]]
            mean, err, tau, dtau = tauint(data, lambda x, y: x * y)
            if tau - 3*dtau > known_tau or \
                tau + 3*dtau < known_tau:
                print tau - known_tau, dtau
                fails += 1
        self.assertGreaterEqual(0, fails)
if __name__ == "__main__":
    main(verbosity = 2)

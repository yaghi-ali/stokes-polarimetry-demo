import unittest
import numpy as np
from analysis import *

class Tests(unittest.TestCase):
    def test_ideal_states(self):
        for s in [np.array([1,0,0,0]),np.array([1,1,0,0]),np.array([1,0,0,1]),np.array([2,.3,-.5,.2])]:
            r,d=reconstruct(M@s);np.testing.assert_allclose(r,s,atol=1e-12)
    def test_gain_scaling(self):
        s=np.array([1,.2,.3,.4]);a,d=reconstruct(M@s);b,e=reconstruct(3*M@s);np.testing.assert_allclose(b,3*a);self.assertAlmostEqual(d,e)
    def test_nonphysical_is_not_hidden(self):
        s,d=reconstruct(np.array([1,0,1,0,1,0]));self.assertGreater(d,1)
    def test_invalid(self):
        for y in [np.zeros(6),np.ones(5),np.array([-1,1,1,1,1,1])]:
            with self.assertRaises(ValueError):reconstruct(y)

import unittest
import numpy as np
import knee.convex_hull as convex_hull


class TestConvexHull(unittest.TestCase):
    def test_ccw_colinear(self):
        a = np.array([-1,0])
        b = np.array([0,0])
        c = np.array([1,0])
        result = convex_hull.ccw(a,b,c)
        desired = 0
        self.assertEqual(result, desired)
    
    def test_ccw_ccw(self):
        a = np.array([1, 0])
        b = np.array([0, 1])
        c = np.array([-1,0])
        result = convex_hull.ccw(a,b,c)
        desired = 0
        self.assertGreater(result, desired)
    
    def test_ccw_cw(self):
        a = np.array([-1,0])
        b = np.array([0, 1])
        c = np.array([1, 0])
        result = convex_hull.ccw(a,b,c)
        desired = 0
        self.assertLess(result, desired)


if __name__ == '__main__':
    unittest.main()
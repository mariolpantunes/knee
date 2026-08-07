
__author__ = 'Mário Antunes'
__version__ = '1.0'
__email__ = 'mario.antunes@ua.pt'
__status__ = 'Development'
__license__ = 'MIT'
__copyright__ = '''
Copyright (c) 2021-2023 Stony Brook University
Copyright (c) 2021-2023 The Research Foundation of SUNY
'''

import math
import unittest

import numpy as np

import kneeliverse.linear_fit as lf
from kneeliverse import metrics


class TestLinearFir(unittest.TestCase):
    
    def test_r2_two(self):
        points = np.array([[0.0, 1.0], [1.0, 5.0]])
        coef = lf.linear_fit_points(points)
        result = lf.linear_r2_points(points, coef)
        desired = 1.0
        self.assertEqual(result, desired)
    
    def test_rmspe(self):
        x = np.array([0,1,2,3,4])
        y = np.array([2,2,2,2,2])
        coef = (2,0)
        result = lf.rmspe(x,y, coef)
        desired = 0.0
        self.assertEqual(result, desired)
    
    def test_angle_00(self):
        coef1 = (0, 0)
        result = lf.angle(coef1, coef1)
        desired = 0.0
        self.assertEqual(result, desired)
    
    def test_angle_01(self):
        coef1 = (0, 0)
        coef2 = (0, 1e10)
        result = math.degrees(math.fabs(lf.angle(coef1, coef2)))
        desired = 89.99999999427042
        self.assertAlmostEqual(result, desired, 2)
    
    def test_rpd(self):
        coef = (0, 1)
        points = np.array([[0.0, 0.0], [1.0, 0.9], [2.0, 1.5], [3,2.25], [4,3.6], [5.0,5.0]])
        result = lf.rpd_points(points, coef)
        desired = 0.116
        self.assertAlmostEqual(result, desired, 2)
    
    def test_r2(self):
        coef = (0, 1)
        points = np.array([[0.0, 0.0], [1.0, 0.9], [2.0, 1.5], [3,2.25], [4,3.6], [5.0,5.0]])
        result = lf.r2_points(points, coef)
        desired = 0.973
        self.assertAlmostEqual(result, desired, 2)
    
    def test_rmspe_points(self):
        # Renamed: this shared its name with the `lf.rmspe` test above, so
        # that one was shadowed and never ran. They cover different functions.
        coef = (0, 1)
        points = np.array([[0.0, 0.0], [1.0, 0.9], [2.0, 1.5], [3,2.25], [4,3.6], [5.0,5.0]])
        result = lf.rmspe_points(points, coef)
        desired = 0.202
        self.assertAlmostEqual(result, desired, 2)


# The fixture every test below shares. `linear_fit` is an ENDPOINT fit, not
# least squares - it draws the line through the first and last point only, as
# its docstring says - so the expected values are hand-computable:
#
#   x = [0,1,2,3,4]  y = [0,1,2,3,5]
#   m = (0-5)/(0-4) = 1.25,  b = 0            -> y_hat = [0, 1.25, 2.5, 3.75, 5]
#   residuals = [0, -0.25, -0.5, -0.75, 0]    -> RSS = 0.875
#   RMSE = sqrt(0.875/5) = sqrt(0.175)
#   SStot (about mean 2.2) = 14.8             -> R2 = 1 - 0.875/14.8
X = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
Y = np.array([0.0, 1.0, 2.0, 3.0, 5.0])
POINTS = np.column_stack((X, Y))
COEF = (0.0, 1.25)
RSS = 0.875


class TestLinearFitCoefficients(unittest.TestCase):
    def test_fits_an_exact_line_exactly(self):
        y = 2.0 * X + 1.0
        b, m = lf.linear_fit(X, y)
        self.assertAlmostEqual(b, 1.0)
        self.assertAlmostEqual(m, 2.0)

    def test_uses_only_the_endpoints(self):
        # Moving an interior point cannot change an endpoint fit. This is the
        # property that distinguishes it from least squares, and the reason
        # the expected values in this file are hand-computable.
        moved = Y.copy()
        moved[2] += 10.0
        self.assertEqual(lf.linear_fit(X, Y), lf.linear_fit(X, moved))

    def test_returns_the_expected_endpoint_coefficients(self):
        b, m = lf.linear_fit(X, Y)
        self.assertAlmostEqual(b, COEF[0])
        self.assertAlmostEqual(m, COEF[1])

    def test_degenerate_x_range_returns_zero_coefficients(self):
        # A vertical run has no y = mx + b to fit; the guard returns (0, 0)
        # rather than dividing by zero.
        self.assertEqual(lf.linear_fit(np.array([2.0, 2.0, 2.0]), np.array([1.0, 5.0, 9.0])), (0, 0))

    def test_points_wrapper_agrees_with_the_xy_form(self):
        self.assertEqual(lf.linear_fit_points(POINTS), lf.linear_fit(X, Y))


class TestLinearTransform(unittest.TestCase):
    def test_applies_the_coefficients(self):
        np.testing.assert_allclose(lf.linear_transform(X, (1.0, 2.0)), 2.0 * X + 1.0)

    def test_points_wrapper_agrees_with_the_xy_form(self):
        np.testing.assert_allclose(lf.linear_transform_points(POINTS, COEF),
                                   lf.linear_transform(X, COEF))

    def test_fit_transform_returns_the_fitted_values(self):
        np.testing.assert_allclose(lf.linear_fit_transform(X, Y), lf.linear_transform(X, COEF))

    def test_fit_transform_vertical_returns_both_axes(self):
        # The vertical form fits f(y) = m*y + b, so it has to hand back the x
        # values as well - the horizontal form returns y_hat alone. Callers
        # unpacking one as the other is a real bug this pins down.
        horizontal = lf.linear_fit_transform_points(POINTS)
        self.assertEqual(np.ndim(horizontal), 1)

        vertical = lf.linear_fit_transform_points(POINTS, vertical=True)
        self.assertEqual(len(vertical), 2)
        self.assertEqual(len(vertical[0]), len(X))
        self.assertEqual(len(vertical[1]), len(X))


class TestResiduals(unittest.TestCase):
    def test_an_exact_fit_has_no_residual(self):
        y = 2.0 * X + 1.0
        self.assertAlmostEqual(lf.linear_residuals(X, y, (1.0, 2.0)), 0.0)
        self.assertAlmostEqual(lf.linear_fit_residuals(X, y), 0.0)
        self.assertAlmostEqual(lf.linear_hv_residuals(X, y), 0.0)

    def test_residuals_are_the_sum_of_squares(self):
        self.assertAlmostEqual(lf.linear_residuals(X, Y, COEF), RSS)

    def test_fit_residuals_fits_first_then_measures(self):
        self.assertAlmostEqual(lf.linear_fit_residuals(X, Y), RSS)

    def test_hv_residuals_take_the_better_orientation(self):
        # It tries both the horizontal and the vertical fit, so it can never
        # be worse than the horizontal one alone.
        self.assertLessEqual(lf.linear_hv_residuals(X, Y), lf.linear_fit_residuals(X, Y))

    def test_points_wrappers_agree_with_the_xy_forms(self):
        self.assertAlmostEqual(lf.linear_residuals_points(POINTS, COEF),
                               lf.linear_residuals(X, Y, COEF))
        self.assertAlmostEqual(lf.linear_fit_residuals_points(POINTS),
                               lf.linear_fit_residuals(X, Y))
        self.assertAlmostEqual(lf.linear_hv_residuals_points(POINTS),
                               lf.linear_hv_residuals(X, Y))


class TestPointErrorMetrics(unittest.TestCase):
    """`*_points` metrics all take (points, coef) and return 0 for an exact
    fit, growing as the fit worsens."""

    def setUp(self):
        self.exact_points = np.column_stack((X, 2.0 * X + 1.0))
        self.exact_coef = (1.0, 2.0)
        self.metrics = (lf.rmse_points, lf.rmsle_points, lf.smape_points, lf.rpd_points)

    def test_an_exact_fit_scores_zero(self):
        for fn in self.metrics:
            with self.subTest(metric=fn.__name__):
                self.assertAlmostEqual(fn(self.exact_points, self.exact_coef), 0.0)

    def test_error_grows_as_the_fit_worsens(self):
        for fn in self.metrics:
            with self.subTest(metric=fn.__name__):
                near = fn(self.exact_points, (1.1, 2.0))
                far = fn(self.exact_points, (3.0, 2.0))
                self.assertLess(near, far)

    def test_rmse_matches_the_hand_computed_value(self):
        self.assertAlmostEqual(lf.rmse_points(POINTS, COEF), math.sqrt(RSS / len(X)))


class TestLinearR2(unittest.TestCase):
    def test_an_exact_fit_scores_one(self):
        y = 2.0 * X + 1.0
        self.assertAlmostEqual(lf.linear_r2(X, y, (1.0, 2.0)), 1.0)

    def test_matches_the_hand_computed_value(self):
        sstot = float(np.sum((Y - Y.mean()) ** 2))     # 14.8
        self.assertAlmostEqual(lf.linear_r2(X, Y, COEF), 1.0 - RSS / sstot)

    def test_adjusted_penalises_relative_to_classic(self):
        classic = lf.linear_r2(X, Y, COEF, metrics.R2.classic)
        adjusted = lf.linear_r2(X, Y, COEF, metrics.R2.adjusted)
        self.assertLess(adjusted, classic)

    def test_points_wrapper_agrees_with_the_xy_form(self):
        self.assertAlmostEqual(lf.linear_r2_points(POINTS, COEF), lf.linear_r2(X, Y, COEF))


class TestDistances(unittest.TestCase):
    """The two distance families differ only when the projection falls
    OUTSIDE the segment - which is exactly when RDP is choosing a split - so
    the distinguishing case matters more than the agreeing one."""

    def setUp(self):
        # Chord from (0,0) to (4,0); interior points sit 1 and 2 above/below.
        self.points = np.array([[0., 0.], [1., 1.], [2., 0.], [3., -2.], [4., 0.]])
        self.expected = np.array([0., 1., 0., 2., 0.])

    def test_perpendicular_distance_to_the_chord(self):
        np.testing.assert_allclose(
            lf.perpendicular_distance_points(self.points, self.points[0], self.points[-1]),
            self.expected)

    def test_shortest_distance_agrees_when_the_projection_is_inside(self):
        np.testing.assert_allclose(
            lf.shortest_distance_points(self.points, self.points[0], self.points[-1]),
            self.expected)

    def test_they_disagree_when_the_projection_falls_outside(self):
        # (5,0) lies ON the infinite line through (0,0)-(1,0) but well past
        # its end: perpendicular distance is 0, shortest clamps to the
        # endpoint and gives 4.
        a, b = np.array([0., 0.]), np.array([1., 0.])
        p = np.array([[5., 0.]])
        self.assertAlmostEqual(float(lf.perpendicular_distance_points(p, a, b)[0]), 0.0)
        self.assertAlmostEqual(float(lf.shortest_distance_points(p, a, b)[0]), 4.0)

    def test_whole_curve_and_index_forms_agree(self):
        np.testing.assert_allclose(lf.perpendicular_distance(self.points), self.expected)
        np.testing.assert_allclose(
            lf.perpendicular_distance_index(self.points, 0, len(self.points) - 1),
            self.expected)


class TestCross2d(unittest.TestCase):
    def test_unit_axes_give_unit_area(self):
        self.assertAlmostEqual(lf.cross2d(np.array([1., 0.]), np.array([0., 1.])), 1.0)

    def test_parallel_vectors_give_zero(self):
        self.assertAlmostEqual(lf.cross2d(np.array([2., 4.]), np.array([1., 2.])), 0.0)

    def test_it_is_antisymmetric(self):
        u, v = np.array([3., 1.]), np.array([1., 2.])
        self.assertAlmostEqual(lf.cross2d(u, v), -lf.cross2d(v, u))


if __name__ == '__main__':
    unittest.main()
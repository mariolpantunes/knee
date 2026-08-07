
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

from kneeliverse import evaluation, metrics, rdp


class TestEvaluation(unittest.TestCase):
    def test_mae_0(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([0,1,2])
        expected = np.array([[0,0], [1,1], [2,2]])
        result = evaluation.mae(points, knees, expected)
        desired = 0.0
        self.assertAlmostEqual(result, desired)
    
    def test_mae_1(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([0,1,2])
        expected = np.array([[1,1], [2,2]])
        result = evaluation.mae(points, knees, expected, evaluation.Strategy.worst)
        desired = 1/3
        self.assertAlmostEqual(result, desired)
    
    def test_mae_2(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([1])
        expected = np.array([[1,1], [2,2]])
        result = evaluation.mae(points, knees, expected)
        desired = 1/2
        self.assertAlmostEqual(result, desired)
    
    def test_mse_0(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([0,1,2])
        expected = np.array([[0,0], [1,1], [2,2]])
        result = evaluation.mse(points, knees, expected)
        desired = 0.0
        self.assertAlmostEqual(result, desired)
    
    def test_mse_1(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([0,1,2])
        expected = np.array([[1,1], [2,2]])
        result = evaluation.mse(points, knees, expected, evaluation.Strategy.worst)
        desired = 1/3
        self.assertAlmostEqual(result, desired)
    
    def test_mse_2(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([1])
        expected = np.array([[1,1], [2,2]])
        result = evaluation.mse(points, knees, expected)
        desired = 1/2
        self.assertAlmostEqual(result, desired)
    
    def test_rmse_0(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([0,1,2])
        expected = np.array([[0,0], [1,1], [2,2]])
        result = evaluation.rmse(points, knees, expected)
        desired = 0.0
        self.assertAlmostEqual(result, desired)
    
    def test_rmse_1(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([0,1,2])
        expected = np.array([[1,1], [2,2]])
        result = evaluation.rmse(points, knees, expected, evaluation.Strategy.worst)
        desired = math.sqrt(1/3)
        self.assertAlmostEqual(result, desired)
    
    def test_rmse_2(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([1])
        expected = np.array([[1,1], [2,2]])
        result = evaluation.rmse(points, knees, expected)
        desired = math.sqrt(1/2)
        self.assertAlmostEqual(result, desired)
    
    def test_rmspe_0(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([0,1,2])
        expected = np.array([[0,0], [1,1], [2,2]])
        result = evaluation.rmspe(points, knees, expected)
        desired = 0.0
        self.assertAlmostEqual(result, desired)
    
    def test_rmspe_1(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([0,1,2])
        expected = np.array([[1,1], [2,2]])
        result = evaluation.rmspe(points, knees, expected, evaluation.Strategy.worst)
        desired = 5773502691896258.0
        self.assertAlmostEqual(result, desired, 3)
    
    def test_rmspe_2(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([1])
        expected = np.array([[1,1], [2,2]])
        result = evaluation.rmspe(points, knees, expected)
        desired = 0.3535533905755961
        self.assertAlmostEqual(result, desired, 3)
    
    def test_cm_0(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([1])
        expected = np.array([[1,1], [2,2]])
        result = evaluation.cm(points, knees, expected)
        desired = np.array([[1,0],[1,1]])
        np.testing.assert_array_equal(result, desired)
    
    def test_cm_1(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([1])
        expected = np.array([[0,0], [2,2]])
        result = evaluation.cm(points, knees, expected, t=.5)
        desired = np.array([[1,0],[1,1]])
        np.testing.assert_array_equal(result, desired)
    
    def test_cm_2(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([0])
        expected = np.array([[1,1], [2,2]])
        result = evaluation.cm(points, knees, expected)
        desired = np.array([[0,1],[2,0]])
        np.testing.assert_array_equal(result, desired)
    
    def test_mcc_0(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([1])
        expected = np.array([[1,1], [2,2]])
        cm = evaluation.cm(points, knees, expected)
        result = evaluation.mcc(cm)
        desired = 0.5
        self.assertAlmostEqual(result, desired, 3)
    
    def test_mcc_1(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([1])
        expected = np.array([[0,0], [2,2]])
        cm = evaluation.cm(points, knees, expected, t=.5)
        result = evaluation.mcc(cm)
        desired = 0.5
        self.assertAlmostEqual(result, desired)
    
    def test_mcc_2(self):
        points = np.array([[0,0], [1,1], [2,2]])
        knees = np.array([0])
        expected = np.array([[1,1], [2,2]])
        cm = evaluation.cm(points, knees, expected)
        result = evaluation.mcc(cm)
        desired = -1.0
        self.assertAlmostEqual(result, desired)

    def test_compute_global_rmse_0(self):
        points = np.array([[0,2], [1,1], [2,0], [3,1], [4,2]])
        reduced = np.array([0,2,4])
        result = evaluation.compute_global_rmse(points, reduced)
        desired = 0.0
        self.assertAlmostEqual(result, desired)
    
    def test_compute_global_rmse_1(self):
        points = np.array([[0,2], [1,1], [2,0], [3,1], [4,2]])
        reduced = np.array([0,1,3,4])
        result = evaluation.compute_global_rmse(points, reduced)
        desired = 0.4472135954999579
        self.assertAlmostEqual(result, desired)
    
    def test_mip_0(self):
        points = np.array([[0,2], [1,1], [2,0], [3,1], [4,2]])
        reduced = np.array([0,2,4])
        mip, _ = evaluation.mip(points, reduced)
        desired = 1.0954451150103321
        self.assertAlmostEqual(mip, desired)

    def test_mip_1(self):
        points = np.array([[0,2], [1,1], [2,0], [3,1], [4,2]])
        reduced = np.array([0,1,3,4])
        mip, _ = evaluation.mip(points, reduced)
        desired = 0.2194530711667088
        self.assertAlmostEqual(mip, desired)

    def test_compute_global_cost_0(self):
        points = np.array([[0, 0], [1, 1], [2, 2], [3, 2], [4, 3], [5, 4]])
        reduced = np.array([0, 2, 3, 5])
        result = evaluation.compute_global_cost(points, reduced)
        desired = 0.0
        self.assertEqual(result, desired)
    
    def test_compute_global_cost_1(self):
        points = np.array([[0, 2], [0, 1], [1/2, 1/2], [1, 0], [2, 0]])
        reduced = np.array([0, 1, 3, 4])
        result = evaluation.compute_global_cost(points, reduced)
        #desired = 0.2857142857142857
        desired = 0.0
        self.assertEqual(result, desired)


class TestComputeMetric(unittest.TestCase):
    """`metrics` defines the Metrics enum and the functions implementing it,
    but nothing mapped one to the other, so every caller spelled the branch
    out by hand - and `compute_global_segment_cost` got it wrong."""

    def setUp(self):
        self.y = np.array([1.0, 2.0, 3.0, 4.0])

    def test_a_perfect_fit_scores_zero_error(self):
        for metric in (metrics.Metrics.rmspe, metrics.Metrics.rmsle,
                       metrics.Metrics.rpd, metrics.Metrics.smape):
            with self.subTest(metric=metric.value):
                self.assertAlmostEqual(evaluation.compute_metric(self.y, self.y, metric), 0.0)

    def test_a_perfect_fit_scores_r2_of_one(self):
        self.assertAlmostEqual(evaluation.compute_metric(self.y, self.y, metrics.Metrics.r2), 1.0)

    def test_error_grows_with_the_residual(self):
        near = evaluation.compute_metric(self.y, self.y + 0.01, metrics.Metrics.rpd)
        far = evaluation.compute_metric(self.y, self.y + 1.00, metrics.Metrics.rpd)
        self.assertLess(near, far)

    def test_accepts_lists_as_well_as_arrays(self):
        self.assertAlmostEqual(
            evaluation.compute_metric([1.0, 2.0], [1.0, 2.0], metrics.Metrics.rpd), 0.0)

    def test_every_enum_member_is_dispatchable(self):
        # A member without an entry would raise KeyError at call time.
        for metric in metrics.Metrics:
            with self.subTest(metric=metric.value):
                self.assertIsInstance(evaluation.compute_metric(self.y, self.y, metric), float)


class TestComputeGlobalSegmentCost(unittest.TestCase):
    """The uncached counterpart to `compute_global_cost`.

    It had no tests and two independent defects that made it raise on every
    call: it unpacked `linear_fit_transform_points` as a 2-tuple when the
    horizontal fit returns y_hat alone, and it then called `compute_cost`
    with three of the four arguments that signature needs, and the wrong
    ones. Neither could survive a single execution.
    """

    @staticmethod
    def _curve():
        y = np.concatenate([np.linspace(1.0, 0.2, 10), np.full(20, 0.2)])
        return np.column_stack((np.arange(len(y), dtype=float), y))

    def test_runs_and_returns_a_cost_per_segment(self):
        points = self._curve()
        reduced, _ = rdp.grdp(points, t=0.01)
        cost, segments = evaluation.compute_global_segment_cost(points, reduced)
        self.assertIsInstance(cost, float)
        self.assertEqual(len(segments), len(reduced) - 1)

    def test_a_straight_line_costs_nothing(self):
        x = np.arange(10, dtype=float)
        points = np.column_stack((x, x))
        cost, segments = evaluation.compute_global_segment_cost(
            points, np.array([0, 9]), metrics.Metrics.rpd)
        self.assertAlmostEqual(cost, 0.0)
        self.assertEqual(len(segments), 1)

    def test_every_metric_is_accepted(self):
        points = self._curve()
        reduced, _ = rdp.grdp(points, t=0.01)
        for metric in metrics.Metrics:
            with self.subTest(metric=metric.value):
                cost, _ = evaluation.compute_global_segment_cost(points, reduced, metric)
                self.assertTrue(math.isfinite(cost))

    def test_a_worse_reduction_costs_more(self):
        # Keeping only the endpoints throws the knee away, so it has to fit
        # worse than a reduction that keeps it.
        points = self._curve()
        reduced, _ = rdp.grdp(points, t=0.01)
        good, _ = evaluation.compute_global_segment_cost(points, reduced, metrics.Metrics.rpd)
        poor, _ = evaluation.compute_global_segment_cost(
            points, np.array([0, len(points) - 1]), metrics.Metrics.rpd)
        self.assertLess(good, poor)


if __name__ == '__main__':
    unittest.main()
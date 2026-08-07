
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


class TestConfusionMatrixScores(unittest.TestCase):
    """`cm` already had tests; the scores derived from it did not.

    All three take the confusion matrix, NOT (points, knees, expected) - an
    easy signature to get wrong, since `mcc` sits beside them and reads the
    same way.
    """

    @staticmethod
    def _curve():
        y = np.concatenate([np.linspace(1.0, 0.3, 8), np.full(12, 0.3)])
        return np.column_stack((np.arange(len(y), dtype=float), y))

    def test_a_perfect_prediction_scores_one(self):
        points = self._curve()
        knees = np.array([3, 7, 12])
        cm = evaluation.cm(points, knees, points[knees])
        self.assertAlmostEqual(evaluation.accuracy(cm), 1.0)
        self.assertAlmostEqual(evaluation.f1score(cm), 1.0)

    def test_accuracy_is_the_diagonal_over_the_total(self):
        cm = np.array([[3, 1], [2, 4]])
        self.assertAlmostEqual(evaluation.accuracy(cm), 7 / 10)

    def test_f1_is_the_harmonic_mean_of_precision_and_recall(self):
        # tp=3, fp=1, fn=2 -> precision 3/4, recall 3/5, F1 = 2PR/(P+R)
        cm = np.array([[3, 1], [2, 4]])
        precision, recall = 3 / 4, 3 / 5
        self.assertAlmostEqual(evaluation.f1score(cm),
                               2 * precision * recall / (precision + recall))

    def test_a_prediction_that_misses_everything_scores_zero(self):
        cm = np.array([[0, 4], [4, 0]])
        self.assertAlmostEqual(evaluation.accuracy(cm), 0.0)
        self.assertAlmostEqual(evaluation.f1score(cm), 0.0)


class TestGetNeighbourhood(unittest.TestCase):
    """All three variants answer the same question - how far left of `a` the
    curve stays straight to within R2 `t` - by different searches, so they
    should broadly agree on a curve with an unambiguous corner."""

    def setUp(self):
        # Straight to index 7, flat after: the neighbourhood left of 10 must
        # stop at the corner rather than running into the slope.
        self.x = np.arange(20, dtype=float)
        self.y = np.concatenate([np.linspace(1.0, 0.3, 8), np.full(12, 0.3)])

    def test_it_stops_at_the_corner(self):
        idx, r2, _ = evaluation.get_neighbourhood(self.x, self.y, 10, 0)
        self.assertEqual(idx, 7)
        self.assertAlmostEqual(r2, 1.0)

    def test_the_fast_variant_agrees(self):
        self.assertEqual(evaluation.get_neighbourhood(self.x, self.y, 10, 0)[0],
                         evaluation.get_neighbourhood_fast(self.x, self.y, 10, 0)[0])

    def test_the_binary_variant_lands_near_the_corner(self):
        # A binary search cannot be exact here, but must not wander far.
        self.assertLessEqual(abs(evaluation.get_neighbourhood_binary(self.x, self.y, 10, 0) - 7), 2)

    def test_points_wrapper_agrees_with_the_xy_form(self):
        points = np.column_stack((self.x, self.y))
        self.assertEqual(evaluation.get_neighbourhood_fast_points(points, 10, 0, 0.9)[0],
                         evaluation.get_neighbourhood_fast(self.x, self.y, 10, 0, 0.9)[0])

    def test_a_threshold_of_one_does_not_raise(self):
        # Regression: `previous_res` was only assigned inside the loop, but
        # returned from the else branch. With t >= 1.0 the loop never runs and
        # this raised NameError.
        idx, r2, slope = evaluation.get_neighbourhood(self.x, self.y, 10, 0, t=1.0)
        self.assertIsInstance(int(idx), int)
        self.assertTrue(math.isfinite(r2))
        self.assertTrue(math.isfinite(slope))


class TestComputePartialCost(unittest.TestCase):
    """The per-segment error that `compute_global_cost` sums. Unlike the
    `metrics` functions it returns a SUM, not a mean, so it grows with the
    length of the segment as well as with the error."""

    def setUp(self):
        self.y = np.array([1.0, 2.0, 3.0, 4.0])

    def test_a_perfect_fit_costs_nothing(self):
        for metric in metrics.Metrics:
            with self.subTest(metric=metric.value):
                self.assertAlmostEqual(
                    evaluation.compute_partial_cost(self.y, self.y, metric), 0.0)

    def test_cost_grows_with_the_residual(self):
        near = evaluation.compute_partial_cost(self.y, self.y + 0.01, metrics.Metrics.rpd)
        far = evaluation.compute_partial_cost(self.y, self.y + 1.00, metrics.Metrics.rpd)
        self.assertLess(near, far)

    def test_it_sums_rather_than_averages(self):
        # Doubling the number of equally-wrong points doubles the cost.
        one = evaluation.compute_partial_cost(self.y, self.y + 1.0, metrics.Metrics.rpd)
        two = evaluation.compute_partial_cost(np.tile(self.y, 2), np.tile(self.y + 1.0, 2),
                                              metrics.Metrics.rpd)
        self.assertAlmostEqual(two, 2 * one)


class TestComputeCost(unittest.TestCase):
    """Folds the per-segment errors into one number. Takes a cache dict - not
    optional - which is what `compute_global_segment_cost` got wrong."""

    def setUp(self):
        y = np.concatenate([np.linspace(1.0, 0.3, 8), np.full(12, 0.3)])
        self.points = np.column_stack((np.arange(len(y), dtype=float), y))

    def test_no_segment_error_costs_nothing(self):
        for metric in (metrics.Metrics.rmsle, metrics.Metrics.rmspe,
                       metrics.Metrics.rpd, metrics.Metrics.smape):
            with self.subTest(metric=metric.value):
                self.assertAlmostEqual(
                    evaluation.compute_cost(self.points, np.zeros(3), metric, {}), 0.0)

    def test_cost_grows_with_the_segment_errors(self):
        near = evaluation.compute_cost(self.points, np.full(3, 0.01), metrics.Metrics.rpd, {})
        far = evaluation.compute_cost(self.points, np.full(3, 1.00), metrics.Metrics.rpd, {})
        self.assertLess(near, far)

    def test_it_never_returns_a_negative_cost(self):
        # r2 is 1-RSS/TSS, which goes negative for a fit worse than the mean;
        # the result is floored at 0.
        self.assertGreaterEqual(
            evaluation.compute_cost(self.points, np.full(3, 1e6), metrics.Metrics.r2, {}), 0.0)

    def test_the_cache_is_populated_and_reused(self):
        cache = {}
        first = evaluation.compute_global_cost(self.points, np.array([0, 8, 19]), cache=cache)
        self.assertTrue(cache)
        second = evaluation.compute_global_cost(self.points, np.array([0, 8, 19]), cache=cache)
        self.assertAlmostEqual(first, second)


class TestAccuracyHeuristics(unittest.TestCase):
    """`accuracy_knee` and `accuracy_trace` score a knee set without ground
    truth, from how well the segments they induce fit."""

    def setUp(self):
        y = np.concatenate([np.linspace(1.0, 0.3, 8), np.full(12, 0.3)])
        self.points = np.column_stack((np.arange(len(y), dtype=float), y))

    def test_both_return_five_finite_components(self):
        for fn in (evaluation.accuracy_knee, evaluation.accuracy_trace):
            with self.subTest(fn=fn.__name__):
                result = fn(self.points, np.array([3, 7, 12]))
                self.assertEqual(len(result), 5)
                self.assertTrue(all(math.isfinite(v) for v in result))

    def test_the_corner_costs_less_than_an_arbitrary_split(self):
        # The tuple is (average_x, average_y, average_slope,
        # average_coefficients, COST) - the last element is a cost, so lower
        # is better. The curve bends at 7, so a knee there explains it better
        # than one dropped in the middle of the flat tail.
        at_corner = evaluation.accuracy_trace(self.points, np.array([7]))[-1]
        in_tail = evaluation.accuracy_trace(self.points, np.array([15]))[-1]
        self.assertLess(at_corner, in_tail)


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
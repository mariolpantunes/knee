
__author__ = 'Mário Antunes'
__version__ = '1.0'
__email__ = 'mario.antunes@ua.pt'
__status__ = 'Development'
__license__ = 'MIT'

import unittest

import numpy as np

from kneeliverse import autoelbow, curvature, dfdt, lmethod, utils


def corner_curve(corner: int, n: int = 30) -> np.ndarray:
    """A straight descent to `corner`, flat after it."""
    y = np.concatenate([np.linspace(1.0, 0.3, corner + 1), np.full(n - corner - 1, 0.3)])
    return np.column_stack((np.arange(len(y), dtype=float), y))


class TestNormalize(unittest.TestCase):
    def test_both_axes_land_on_the_unit_interval(self):
        points = np.column_stack((np.arange(10, dtype=float) * 3.0 + 5.0,
                                  np.linspace(100.0, 20.0, 10)))
        out = autoelbow.normalize(points)
        for axis in (0, 1):
            self.assertAlmostEqual(out[:, axis].min(), 0.0)
            self.assertAlmostEqual(out[:, axis].max(), 1.0)

    def test_it_preserves_shape_and_order(self):
        points = np.column_stack((np.arange(10, dtype=float), np.linspace(5.0, 1.0, 10)))
        out = autoelbow.normalize(points)
        self.assertEqual(out.shape, points.shape)
        self.assertTrue(np.all(np.diff(out[:, 1]) < 0))

    def test_a_degenerate_axis_becomes_zeros(self):
        # Every y identical: the span is 0 and must not be divided by.
        points = np.column_stack((np.arange(5, dtype=float), np.full(5, 7.0)))
        out = autoelbow.normalize(points)
        np.testing.assert_allclose(out[:, 1], np.zeros(5))

    def test_it_is_scale_and_offset_invariant(self):
        points = np.column_stack((np.arange(10, dtype=float), np.linspace(1.0, 0.2, 10)))
        shifted = points.copy()
        shifted[:, 1] = shifted[:, 1] * 1000.0 + 42.0
        np.testing.assert_allclose(autoelbow.normalize(points),
                                   autoelbow.normalize(shifted))


class TestEnforceMonotonicity(unittest.TestCase):
    def test_an_already_monotone_curve_is_untouched(self):
        y = np.linspace(1.0, 0.0, 10)
        np.testing.assert_allclose(autoelbow.enforce_monotonicity(y, decreasing=True), y)

    def test_it_removes_an_upward_blip_from_a_falling_curve(self):
        y = np.array([1.0, 0.8, 0.9, 0.4, 0.2])       # 0.9 runs the wrong way
        out = autoelbow.enforce_monotonicity(y, decreasing=True)
        self.assertLessEqual(out[2], out[1])

    def test_it_removes_a_downward_blip_from_a_rising_curve(self):
        y = np.array([0.0, 0.4, 0.3, 0.8, 1.0])
        out = autoelbow.enforce_monotonicity(y, decreasing=False)
        self.assertGreaterEqual(out[2], out[1])

    def test_it_does_not_mutate_the_input(self):
        y = np.array([1.0, 0.8, 0.9, 0.4, 0.2])
        before = y.copy()
        autoelbow.enforce_monotonicity(y, decreasing=True)
        np.testing.assert_array_equal(y, before)


class TestScore(unittest.TestCase):
    def test_one_score_per_point(self):
        points = corner_curve(10)
        self.assertEqual(len(autoelbow.score(points)), len(points))

    def test_scores_are_finite_and_non_negative(self):
        scores = autoelbow.score(corner_curve(10))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertTrue(np.all(scores >= 0.0))

    def test_the_peak_sits_at_the_corner(self):
        scores = autoelbow.score(corner_curve(10))
        self.assertEqual(int(np.argmax(scores)), 10)


class TestAutoElbowKnee(unittest.TestCase):
    def test_it_finds_the_corner(self):
        for corner in (5, 10):
            with self.subTest(corner=corner):
                self.assertEqual(autoelbow.knee(corner_curve(corner)), corner)

    def test_it_stays_close_on_later_corners(self):
        # The monotonicity cleaning pulls the estimate slightly left on a
        # long descent; the paper reports the same tolerance against ground
        # truth, so it is the method's behaviour rather than a defect.
        for corner in (15, 20):
            with self.subTest(corner=corner):
                self.assertLessEqual(abs(autoelbow.knee(corner_curve(corner)) - corner), 2)

    def test_it_agrees_with_the_other_detectors(self):
        points = corner_curve(10)
        expected = autoelbow.knee(points)
        for name, fn in (('curvature', curvature.knee), ('dfdt', dfdt.knee),
                         ('lmethod', lmethod.knee)):
            with self.subTest(detector=name):
                self.assertLessEqual(abs(int(fn(points)) - expected), 1)

    def test_it_handles_all_four_orientations(self):
        x = np.arange(30, dtype=float)
        shapes = {
            (utils.Direction.Decreasing, utils.Concavity.Counterclockwise): np.exp(-x / 6),
            (utils.Direction.Increasing, utils.Concavity.Counterclockwise): np.exp(-(29 - x) / 6),
            (utils.Direction.Increasing, utils.Concavity.Clockwise): 1 - np.exp(-x / 6),
            (utils.Direction.Decreasing, utils.Concavity.Clockwise): 1 - np.exp(-(29 - x) / 6),
        }
        for orientation, y in shapes.items():
            with self.subTest(orientation=[str(o) for o in orientation]):
                points = np.column_stack((x, y))
                self.assertEqual(utils.detect_orientation(points), orientation)
                idx = autoelbow.knee(points)
                self.assertGreater(idx, 0)
                self.assertLess(idx, len(points) - 1)

    def test_it_needs_no_parameters(self):
        # The distinguishing property: unlike kneedle or lmethod there is
        # nothing to tune, so the answer is a function of the curve alone.
        import inspect
        signature = inspect.signature(autoelbow.knee)
        self.assertEqual(list(signature.parameters), ['points'])

    def test_a_flat_curve_has_no_knee(self):
        x = np.arange(30, dtype=float)
        self.assertEqual(autoelbow.knee(np.column_stack((x, np.ones(30)))), 0)

    def test_a_curve_too_short_to_bend(self):
        self.assertEqual(autoelbow.knee(np.array([[0.0, 1.0], [1.0, 0.0]])), 0)

    def test_the_index_is_inside_the_curve(self):
        points = corner_curve(10)
        idx = autoelbow.knee(points)
        self.assertGreaterEqual(idx, 0)
        self.assertLess(idx, len(points))

    def test_it_is_invariant_to_rescaling_the_axes(self):
        points = corner_curve(10)
        scaled = points.copy()
        scaled[:, 0] = scaled[:, 0] * 7.0 + 3.0
        scaled[:, 1] = scaled[:, 1] * 250.0 - 11.0
        self.assertEqual(autoelbow.knee(points), autoelbow.knee(scaled))

    def test_multi_knee_returns_valid_indexes(self):
        knees = np.asarray(autoelbow.multi_knee(corner_curve(10)))
        self.assertTrue(np.all(knees >= 0))
        self.assertTrue(np.all(knees < len(corner_curve(10))))


class TestAutoElbowDeterminism(unittest.TestCase):
    def test_the_choice_is_invariant_to_last_bit_noise(self):
        base = corner_curve(10)
        rng = np.random.default_rng(0)
        picks = set()
        for _ in range(100):
            points = base.copy()
            points[:, 1] += rng.uniform(-1e-15, 1e-15, len(points))
            picks.add(autoelbow.knee(points))
        self.assertEqual(len(picks), 1)

    def test_repeated_calls_agree(self):
        points = corner_curve(10)
        self.assertEqual(autoelbow.knee(points), autoelbow.knee(points))


if __name__ == '__main__':
    unittest.main()

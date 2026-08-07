
__author__ = 'Mário Antunes'
__version__ = '1.0'
__email__ = 'mario.antunes@ua.pt'
__status__ = 'Development'
__license__ = 'MIT'
__copyright__ = '''
Copyright (c) 2021-2023 Stony Brook University
Copyright (c) 2021-2023 The Research Foundation of SUNY
'''

import unittest
import warnings

import numpy as np

from kneeliverse import kneedle, utils


class TestOrientationVocabulary(unittest.TestCase):
    """`Direction` and `Concavity` began life in `kneedle` and are used as
    `kneedle.Direction` by the demos and examples, so moving them here must
    not break that spelling."""

    def test_the_enums_are_reachable_from_kneedle(self):
        self.assertIs(kneedle.Direction, utils.Direction)
        self.assertIs(kneedle.Concavity, utils.Concavity)

    def test_member_order_is_preserved(self):
        # The demos pass list(Concavity) as argparse `choices`, so the order
        # is part of their command-line surface.
        self.assertEqual([c.value for c in utils.Concavity],
                         ['counter-clockwise', 'clockwise'])
        self.assertEqual([d.value for d in utils.Direction],
                         ['increasing', 'decreasing'])

    def test_members_stringify_to_their_value(self):
        self.assertEqual(str(utils.Direction.Increasing), 'increasing')
        self.assertEqual(str(utils.Concavity.Clockwise), 'clockwise')


class TestNormalize(unittest.TestCase):
    def test_both_axes_land_on_the_unit_interval(self):
        points = np.column_stack((np.arange(10, dtype=float) * 3.0 + 5.0,
                                  np.linspace(100.0, 20.0, 10)))
        out = utils.normalize(points)
        for axis in (0, 1):
            self.assertAlmostEqual(out[:, axis].min(), 0.0)
            self.assertAlmostEqual(out[:, axis].max(), 1.0)

    def test_it_preserves_shape_and_order(self):
        points = np.column_stack((np.arange(10, dtype=float), np.linspace(5.0, 1.0, 10)))
        out = utils.normalize(points)
        self.assertEqual(out.shape, points.shape)
        self.assertTrue(np.all(np.diff(out[:, 1]) < 0))

    def test_a_degenerate_axis_becomes_zeros(self):
        # Every y identical: the span is 0 and must not be divided by.
        points = np.column_stack((np.arange(5, dtype=float), np.full(5, 7.0)))
        out = utils.normalize(points)
        np.testing.assert_allclose(out[:, 1], np.zeros(5))

    def test_it_is_scale_and_offset_invariant(self):
        points = np.column_stack((np.arange(10, dtype=float), np.linspace(1.0, 0.2, 10)))
        shifted = points.copy()
        shifted[:, 1] = shifted[:, 1] * 1000.0 + 42.0
        np.testing.assert_allclose(utils.normalize(points),
                                   utils.normalize(shifted))


class TestNormalizeGuardsDegenerateAxes(unittest.TestCase):
    """The reason this lives here rather than inside one detector.

    `kneedle.knee` guarded a constant axis and `kneedle.knees`, a few lines
    away in the same module, did not - so a flat curve gave a knee from one
    and a RuntimeWarning plus NaNs from the other. One implementation, one
    policy.
    """

    def setUp(self):
        self.flat = np.column_stack((np.arange(20, dtype=float), np.full(20, 0.5)))

    def test_a_constant_axis_maps_to_zeros(self):
        np.testing.assert_allclose(utils.normalize(self.flat)[:, 1], np.zeros(20))

    def test_it_produces_no_warning_and_no_nan(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            out = utils.normalize(self.flat)
        self.assertFalse(np.any(np.isnan(out)))

    def test_both_kneedle_entry_points_survive_a_flat_curve(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            kneedle.knee(self.flat)
            kneedle.knees(self.flat)


class TestSpan(unittest.TestCase):
    def test_it_returns_the_extent_of_each_axis(self):
        points = np.column_stack((np.arange(11, dtype=float), np.linspace(2.0, 7.0, 11)))
        dx, dy = utils.span(points)
        self.assertAlmostEqual(dx, 10.0)
        self.assertAlmostEqual(dy, 5.0)

    def test_it_is_never_negative(self):
        # A descending curve ends below where it started, but its extent is
        # still a magnitude.
        points = np.column_stack((np.arange(5, dtype=float), np.linspace(9.0, 1.0, 5)))
        dx, dy = utils.span(points)
        self.assertGreaterEqual(dx, 0.0)
        self.assertAlmostEqual(dy, 8.0)

    def test_a_constant_axis_spans_nothing(self):
        points = np.column_stack((np.arange(5, dtype=float), np.full(5, 3.0)))
        self.assertAlmostEqual(utils.span(points)[1], 0.0)

    def test_it_returns_plain_floats(self):
        points = np.column_stack((np.arange(5, dtype=float), np.linspace(1.0, 2.0, 5)))
        for value in utils.span(points):
            self.assertIsInstance(value, float)


class TestDetectOrientation(unittest.TestCase):
    """The four shapes a knee/elbow graph can take. Direction is the sign of
    the endpoint slope; concavity is which side of that chord the curve
    spends most of itself on."""

    def setUp(self):
        self.x = np.arange(30, dtype=float)

    def _points(self, y):
        return np.column_stack((self.x, y))

    def test_left_elbow_is_decreasing_and_convex(self):
        d, c = utils.detect_orientation(self._points(np.exp(-self.x / 6)))
        self.assertIs(d, utils.Direction.Decreasing)
        self.assertIs(c, utils.Concavity.Counterclockwise)

    def test_right_elbow_is_increasing_and_convex(self):
        d, c = utils.detect_orientation(self._points(np.exp(-(29 - self.x) / 6)))
        self.assertIs(d, utils.Direction.Increasing)
        self.assertIs(c, utils.Concavity.Counterclockwise)

    def test_left_knee_is_increasing_and_concave(self):
        d, c = utils.detect_orientation(self._points(1 - np.exp(-self.x / 6)))
        self.assertIs(d, utils.Direction.Increasing)
        self.assertIs(c, utils.Concavity.Clockwise)

    def test_right_knee_is_decreasing_and_concave(self):
        d, c = utils.detect_orientation(self._points(1 - np.exp(-(29 - self.x) / 6)))
        self.assertIs(d, utils.Direction.Decreasing)
        self.assertIs(c, utils.Concavity.Clockwise)

    def test_a_straight_line_is_flagged_convex(self):
        # Every residual is 0, so the vote is 0 and the tie resolves one way
        # deterministically rather than on the sign of noise.
        d, c = utils.detect_orientation(self._points(2.0 * self.x + 1.0))
        self.assertIs(d, utils.Direction.Increasing)
        self.assertIs(c, utils.Concavity.Counterclockwise)

    def test_reversing_the_curve_flips_the_direction_not_the_concavity(self):
        convex_down = self._points(np.exp(-self.x / 6))
        convex_up = self._points(np.exp(-self.x / 6)[::-1])
        self.assertIs(utils.detect_orientation(convex_down)[0], utils.Direction.Decreasing)
        self.assertIs(utils.detect_orientation(convex_up)[0], utils.Direction.Increasing)
        self.assertIs(utils.detect_orientation(convex_down)[1],
                      utils.detect_orientation(convex_up)[1])

    def test_it_survives_a_corrupted_sample(self):
        # The reason this sums over every point instead of testing one. A
        # midpoint-only test - as published with AutoElbow - disagreed with
        # this on 168 of 200 spiked curves; summing keeps the classification.
        rng = np.random.default_rng(0)
        for _ in range(100):
            y = np.exp(-self.x / 6).copy()
            y[len(self.x) // 2] += rng.uniform(0.3, 0.9)
            d, c = utils.detect_orientation(self._points(y))
            self.assertIs(d, utils.Direction.Decreasing)
            self.assertIs(c, utils.Concavity.Counterclockwise)

    def test_classification_is_invariant_to_last_bit_noise(self):
        base = np.exp(-self.x / 6)
        rng = np.random.default_rng(0)
        seen = set()
        for _ in range(100):
            y = base + rng.uniform(-1e-15, 1e-15, len(base))
            seen.add(utils.detect_orientation(self._points(y)))
        self.assertEqual(len(seen), 1)


if __name__ == '__main__':
    unittest.main()

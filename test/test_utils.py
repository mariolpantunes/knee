
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

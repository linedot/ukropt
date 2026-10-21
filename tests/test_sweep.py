"""ukropt-sweep's parts that need no hardware"""

import unittest

from ukropt.sweep import blis_stub, combinations


class test_sweep(unittest.TestCase):

    def test_combinations(self):
        grid = {"a": [1, 2], "b": ["x", None, "y"]}
        every = combinations(grid, None, 1)
        self.assertEqual(6, len(every))
        self.assertIn({"a": 2, "b": None}, every)
        # a sample: that many, reproducibly
        self.assertEqual(combinations(grid, 3, 7), combinations(grid, 3, 7))
        self.assertEqual(3, len(combinations(grid, 3, 7)))

    def test_stub(self):
        self.assertIn("dim_t", blis_stub())


if __name__ == "__main__":
    unittest.main()

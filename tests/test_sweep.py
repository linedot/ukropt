"""ukropt-sweep's parts that need no hardware"""

import subprocess
import sys
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


class test_options(unittest.TestCase):
    """
    An option a mode doesn't use, or two that don't go together: the sweep
    stops and says why, before reading the family file - the modes and
    their options in its help
    """
    def last(self, *args):
        r = subprocess.run([sys.executable, "-m", "ukropt.sweep", *args, "--out", "/nonexistent"],
                           capture_output=True, text=True, timeout=120)
        self.assertNotEqual(0, r.returncode)
        return (r.stderr or r.stdout).strip().splitlines()[-1]

    def test_refused(self):
        for args, said in (
                (["x/dgemm.yaml", "--family", "f", "--size", "1x4", "--levels", "L1=8K"],
                 "--levels: not in gemm mode"),
                (["x/dpackm.yaml", "--family", "f", "--size", "4", "--levels", "L1=8K", "--loop", "64"],
                 "--levels or --loop"),
                (["x/dpackm.yaml", "--family", "f", "--blis", "cdim=4", "--size", "4"], "--size: not in blis mode"),
                (["x/dpackm.yaml", "--family", "f", "--blis", "cdim=4", "--k", "64"], "--k: not in blis mode"),
                (["x/dpackm.yaml", "--family", "f", "--blis", "cdim=4", "--cpu", "1,2"], "--cpu: --blis"),
                (["x/dgemm.yaml", "--family", "f", "--size", "1x4", "--any", "--case", "vec"],
                 "--case: a gemm family has no cases"),
                (["x/dpackm.yaml", "--family", "f", "--size", "4", "--levels", "L1=8K", "--rank", "max"],
                 "with --levels no line is fitted"),
                (["x/dgemm.yaml", "--family", "f", "--size", "1x4", "--event", "X"], "with --perf-lib")):
            with self.subTest(said=said):
                self.assertIn(said, self.last(*args))

    def test_help(self):
        r = subprocess.run([sys.executable, "-m", "ukropt.sweep", "--help"], capture_output=True, text=True,
                           timeout=120)
        self.assertIn("options by mode - gemm | any-kernel | BLIS-like", r.stdout)



if __name__ == "__main__":
    unittest.main()

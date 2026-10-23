"""
ukropt-pairs and ukropt-fmaload build for RVV, NEON, SVE and x86 - where
the compilers are there: the tools' share of asmgen's 0.6 rework checks
(asmgen's docs/V06_REWORK.md, phase 0)
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from parameterized import parameterized

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
TARGETS = {
    "rvv": ["riscv64-linux-gnu-gcc", "-O2 -march=rv64gcv -static"],
    "neon": ["aarch64-linux-gnu-gcc", "-O2 -static"],
    "sve": ["aarch64-linux-gnu-gcc", "-O2 -march=armv8.2-a+sve -static"],
    "avx256": ["cc", "-O2"],
}
CASES = [
    *[(f"pairs_{isa}", "pairs", isa, []) for isa in TARGETS],
    ("pairs_rvv_memory", "pairs", "rvv", ["--experiments", os.path.join(EXAMPLES, "pairs-memory.yaml"),
                                          "--vector-bytes", "32"]),
    ("pairs_rvv_fetch", "pairs", "rvv", ["--experiments", os.path.join(EXAMPLES, "pairs-fetch.yaml"),
                                         "--loop-offset", "0,4"]),
    ("fmaload_rvv", "fmaload", "rvv", ["--dt", "FP64,FP32,FP16", "--form", "vf,vv"]),
    *[(f"fmaload_{isa}", "fmaload", isa, ["--dt", "FP64,FP32", "--form", "vf,vv"])
      for isa in ("neon", "sve", "avx256")],
]


class test_builds(unittest.TestCase):
    @parameterized.expand(CASES)
    def test_builds(self, _name, tool, isa, extra):
        cc, cflags = TARGETS[isa]
        if not shutil.which(cc):
            self.skipTest(f"no {cc}")
        with tempfile.TemporaryDirectory() as out:
            r = subprocess.run([sys.executable, "-m", f"ukropt.{tool}", isa, "--build-only",
                                "--out", out, "--cc", cc, f"--cflags={cflags}", *extra],
                               capture_output=True, text=True, timeout=900)
            self.assertEqual(0, r.returncode, r.stderr[-1500:] or r.stdout[-1500:])
            self.assertTrue(any(f.startswith("ukr") and os.access(os.path.join(out, f), os.X_OK)
                                for f in os.listdir(out)), f"{tool} {isa}: no binary in {os.listdir(out)}")


if __name__ == "__main__":
    unittest.main()

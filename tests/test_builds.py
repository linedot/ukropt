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
                                         "--loop-offset", "0,4,6"]),
    # the loop's head past a 64-byte boundary by nops - AArch64's 4 bytes,
    # RISC-V's c.nop for 2 mod 4 (asmgen's align)
    ("pairs_neon_fetch", "pairs", "neon", ["--experiments", os.path.join(EXAMPLES, "pairs-fetch.yaml"),
                                           "--loop-offset", "0,8"]),
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



def ukrgen_configs():
    """
    ukrgen's ukrconfig directory, beside its source - None where it isn't
    """
    try:
        import ukrgen
    except ImportError:
        return None
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(ukrgen.__file__))))
    path = os.path.join(root, "ukrconfig")
    return path if os.path.isdir(path) else None


class test_sweep_builds(unittest.TestCase):
    """
    ukropt-sweep builds a few variants: its C driver compiles (asmgen's and
    ukrgen's helpers in it once each) and its kernels assemble without
    warnings
    """
    def test_rvv_vm(self):
        cc, cflags = TARGETS["rvv"]
        configs = ukrgen_configs()
        if not shutil.which(cc) or configs is None:
            self.skipTest("no RISC-V compiler or no ukrgen checkout")
        with tempfile.TemporaryDirectory() as out:
            r = subprocess.run([sys.executable, "-m", "ukropt.sweep",
                                os.path.join(configs, "rvv", "fma_vf", "dgemm.yaml"),
                                "--family", "vm", "--size", "2x14", "--sample", "2",
                                "--grid", os.path.join(EXAMPLES, "grid-k1-vm.yaml"),
                                "--build-only", "--out", out, "--cc", cc, f"--cflags={cflags}"],
                               capture_output=True, text=True, timeout=1800)
            self.assertEqual(0, r.returncode, r.stderr[-2000:] or r.stdout[-2000:])
            self.assertNotIn("static static", r.stderr)
            self.assertNotIn("Warning", r.stderr, r.stderr[-1500:])



def has_avx2():
    try:
        return any("avx2" in line.split() for line in open("/proc/cpuinfo") if line.startswith("flags"))
    except OSError:
        return False


class test_anykernel(unittest.TestCase):
    """
    ukropt-sweep's any-kernel mode: a packm kernel called through its own
    symbol, its arguments from its layouts and sizes, timed at working sets
    filling levels or at loop counts - and a gemm kernel the same way
    """
    def sweep(self, *args):
        configs = ukrgen_configs()
        if configs is None:
            self.skipTest("no ukrgen checkout")
        with tempfile.TemporaryDirectory() as out:
            r = subprocess.run([sys.executable, "-m", "ukropt.sweep", os.path.join(configs, args[0]), *args[1:],
                                "--out", out, "--reps", "2", "--iterations", "4"],
                               capture_output=True, text=True, timeout=1800)
            self.assertEqual(0, r.returncode, r.stderr[-2000:] or r.stdout[-2000:])
            return r.stdout, out

    def test_packm_levels(self):
        if not (shutil.which("gcc") and has_avx2()):
            self.skipTest("needs gcc and an AVX2 CPU")
        said, _ = self.sweep(os.path.join("avx2", "fma_bc_blocks", "dpackm.yaml"), "--family", "pk",
                             "--case", "trans", "--size", "4", "--levels", "L1=8K,L2=64K",
                             "--cflags=-O2 -mavx2 -mfma")
        self.assertIn("per level (L1, L2)", said)
        self.assertRegex(said, r"L1: +[\d.]+ \(\d+\) +[\d.]+ B \| L2:")

    def test_gemm_any(self):
        if not (shutil.which("gcc") and has_avx2()):
            self.skipTest("needs gcc and an AVX2 CPU")
        said, _ = self.sweep(os.path.join("avx2", "fma_bc_blocks", "dgemm.yaml"), "--family", "vm",
                             "--size", "1x4", "--any", "--loop", "16,32,48", "--cflags=-O2 -mavx2 -mfma")
        self.assertIn("per loop step", said)

    def test_packm_builds_rvv(self):
        cc, cflags = TARGETS["rvv"]
        if not shutil.which(cc):
            self.skipTest("no RISC-V compiler")
        _, _ = self.sweep(os.path.join("rvv", "fma_vf", "dpackm.yaml"), "--family", "pk", "--case", "trans",
                          "--size", "3", "--levels", "L1=16K", "--build-only", "--cc", cc, f"--cflags={cflags}")

if __name__ == "__main__":
    unittest.main()


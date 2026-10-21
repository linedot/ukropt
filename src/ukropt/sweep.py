#!/usr/bin/env python3
"""
ukropt-sweep: a gemm microkernel's family settings searched by measuring them.

Each combination of the grid's values - every one, or a random sample - is a
kernel built by ukrgen (the family's settings with the combination's over
them); all of them go into one benchmark binary, which times each kernel
through its BLIS wrapper with A and B packed in L1 at two values of k - the
cycles a k step takes and the cycles a call takes outside the loop - and the
settings are ranked. Cycles from Linux's perf_event_open, else wall-clock time - or, with
--perf-lib, from github.com/linedot/performance_counters (perf, PAPI, M1).

  ukropt-sweep ukrconfig/rvv/fma_vf/dgemm.yaml --family vm --size 2x14 \\
      --grid grid.yaml [--sample 32] [--cc gcc] [--k 64,256] [--build-only]
      [--remote k1:/tmp --cc riscv64-linux-gnu-gcc --cflags "-O2 -march=rv64gcv -static"]

grid.yaml - family keys or flow parameters to lists of values:
  sched-cycle-slots: ["2:vl|vs|vf,sl,i", "3:vf|vs|vf,vl|vs|vf,sl,sl,i;vf-sl"]
  buckets: [null, "vf:5", "vf:8"]
  schedule: [["RAW:data::::13"], ["RAW:data:freg:::7", "RAW:data:vreg:::7"]]
"""

import argparse
import copy
import itertools
import os
import random
import re
import subprocess
import sys
import tempfile

import yaml

# ukrgen and asmgen: whatever is installed in this environment
try:
    import ukrgen  # noqa: F401
except ImportError:
    sys.exit("ukropt-sweep needs ukrgen (and asmgen) installed in this environment")

from ukrgen.injectors.blis.families import configuration, family, build_gemm, key_params  # noqa: E402
from ukrgen.injectors.blis.cgen import gemm_wrapper, asm_source, vl_helpers, c_expr      # noqa: E402
from ukrgen.flow.isa_map import make_generator                                          # noqa: E402

DRIVER = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <sys/wait.h>
#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#endif
static int cycles_fd = -1;
static void counter_open(void)
{
#ifdef __linux__
    struct perf_event_attr a; memset(&a, 0, sizeof a);
    a.type = PERF_TYPE_HARDWARE; a.size = sizeof a; a.config = PERF_COUNT_HW_CPU_CYCLES;
    a.exclude_kernel = 1; a.exclude_hv = 1;
    cycles_fd = (int)syscall(SYS_perf_event_open, &a, 0, -1, -1, 0);
#endif
}
#ifdef USE_PCLIB
int pc_init(const char*); void pc_tic(void); unsigned long long pc_toc(void);
#endif
static double now(void)
{
    if (cycles_fd >= 0) { uint64_t c = 0; if (read(cycles_fd, &c, sizeof c) == sizeof c) return (double)c; }
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec;
}
typedef void (*ukr_t)(dim_t, dim_t, dim_t, const void*, const void*, const void*, const void*, void*,
                      inc_t, inc_t, const auxinfo_t*, const cntx_t*);
static double best(ukr_t f, dim_t mr, dim_t nr, dim_t k, const WT* a, const WT* b, WT* c, int reps)
{
    WT alpha = 1, beta = 1; double t = 1e300;
    for (int r = 0; r < reps; ++r) {
#ifdef USE_PCLIB
        pc_tic();
        for (int i = 0; i < 64; ++i) f(mr, nr, k, &alpha, a, b, &beta, c, 1, mr, NULL, NULL);
        double d = (double)pc_toc() / 64;
#else
        double t0 = now();
        for (int i = 0; i < 64; ++i) f(mr, nr, k, &alpha, a, b, &beta, c, 1, mr, NULL, NULL);
        double d = (now() - t0) / 64;
#endif
        if (d < t) t = d;
    }
    return t;
}
int main(int argc, char** argv)
{
    int nk = argc - 1, ks[64];
    for (int i = 0; i < nk && i < 64; ++i) ks[i] = atoi(argv[i + 1]);
    /* each variant measured in a child of its own (a counter of its own):
       a variant that crashes is reported, the others still run */
#ifdef USE_PCLIB
#define OPEN_COUNTER() pc_init(PC_EVENT)
#else
#define OPEN_COUNTER() counter_open()
    counter_open();
    if (cycles_fd >= 0) { printf("# unit cycles\n"); close(cycles_fd); cycles_fd = -1; }
    else printf("# unit ns\n");
#endif
    int kmax = 1; for (int i = 0; i < nk; ++i) if (ks[i] > kmax) kmax = ks[i];
    /* panels as large as the largest k needs (up to 256 elements a step) */
    size_t na = (size_t)kmax * 256 + 4096;
    WT* a = aligned_alloc(256, na * sizeof(WT)); WT* b = aligned_alloc(256, na * sizeof(WT));
    WT* c = aligned_alloc(256, 4096 * sizeof(WT));
    for (size_t i = 0; i < na; ++i) { a[i] = (WT)((i % 7) * 0.01); b[i] = (WT)((i % 5) * 0.01); }
    memset(c, 0, 4096 * sizeof(WT));
#ifdef USE_PCLIB
    printf("# unit %s\n", PC_EVENT);
#endif
    fflush(stdout);
VARIANTS
    return 0;
}
"""


def blis_stub() -> str:
    """
    The BLIS declarations the kernels' wrappers compile against: ukrgen's
    own where it ships them, else the copy here
    """
    from importlib.resources import files
    try:
        return (files("ukrgen.injectors.blis") / "blis_stub.h").read_text()
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        return (files("ukropt") / "data" / "blis_stub.h").read_text()


def combinations(grid: dict, sample: int | None, seed: int) -> list[dict]:
    keys = list(grid)
    every = [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]
    if sample and sample < len(every):
        random.Random(seed).shuffle(every)
        every = every[:sample]
    return every


def build_variant(job):
    """
    One variant, in a worker process: the kernel's assembly, BLIS wrapper and
    vector-length helpers as text, or why it didn't build
    """
    i, settings, config, family_name, m, n = job
    dtype = "FP32" if os.path.basename(config).startswith("s") else "FP64"
    cfg = configuration.from_files({("gemm", dtype): config})
    base = next(f for f in cfg.families if f.name == family_name)
    fam = copy.deepcopy(base)
    fam.name = f"{base.name}{i}"
    fam.sizes = {"m": [m], "n": [n]}
    # a family key maps to its flow parameter as in a family file, anything
    # else is a flow parameter as it is; null removes the option
    for k, v in settings.items():
        mapped = key_params("gemm", {k: v if v is not None else "x"})
        names = list(mapped) or [k]
        if v is None:
            for name in names:
                fam.params.pop(name, None)
        elif mapped:
            fam.params.update(mapped)
        else:
            fam.params[k] = v
    try:
        ok, skipped = build_gemm(cfg.isa, cfg.name, fam)
    except Exception as err:            # a variant's failure is its own
        return {"i": i, "settings": settings, "error": str(err)[:200]}
    if not ok:
        return {"i": i, "settings": settings, "error": skipped[0][2][:200]}
    k = ok[0]
    ctype = "double" if "FP64" == dtype else "float"
    return {"i": i, "settings": settings, "asm": asm_source(k), "wrapper": gemm_wrapper(k),
            "helpers": vl_helpers([k], make_generator(cfg.isa)), "symbol": k.symbol,
            "mr": c_expr(k.mr, ctype, k.vl_fn), "nr": c_expr(k.nr, ctype, k.vl_fn)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="a gemm family file (ukrconfig/<isa>/<config>/dgemm.yaml)")
    p.add_argument("--family", required=True)
    p.add_argument("--size", required=True, help="MxN in the family's units: 2x14")
    p.add_argument("--grid", required=True, help="YAML: family keys or flow parameters to lists of values")
    p.add_argument("--sample", type=int, default=None, help="a random sample of the combinations")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cc", default="gcc")
    p.add_argument("--cflags", default="-O2")
    p.add_argument("--k", default="16,32,48,64,80",
                   help="values of k to time, a line fitted through them - keep A's and B's panels "
                        "((MR+NR) x k elements) in L1")
    p.add_argument("-j", "--jobs", type=int, default=os.cpu_count(), help="variants built in parallel")
    p.add_argument("--align", default="0",
                   help="byte offsets from a 64-byte boundary to place each kernel at (multiples of 4: "
                        "0,4,8,16,32): each variant measured at each, best and worst reported - how much "
                        "of a difference is placement")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--out", default="ukrsweep")
    p.add_argument("--build-only", action="store_true", help="build the binary, run it elsewhere")
    p.add_argument("--remote", default=None,
                   help="HOST:DIR - copy the binary there (scp) and run it (ssh): generate and "
                        "cross-compile here, measure on the target")
    p.add_argument("--perf-lib", default=None,
                   help="a clone of github.com/linedot/performance_counters: its counters instead")
    p.add_argument("--perf-backend", default="perf", help="perf, papi, m1 or simple")
    p.add_argument("--event", default="CYCLES", help="the counter (PAPI: PAPI_TOT_CYC)")
    p.add_argument("--cxx", default="g++")
    args = p.parse_args()

    dtype = "FP32" if os.path.basename(args.config).startswith("s") else "FP64"
    cfg = configuration.from_files({("gemm", dtype): args.config})
    base = next(f for f in cfg.families if f.name == args.family)
    m, n = (int(x) for x in args.size.lower().split("x"))
    grid = yaml.safe_load(open(args.grid))
    variants = combinations(grid, args.sample, args.seed)
    jobs = [(i, v, args.config, args.family, m, n) for i, v in enumerate(variants)]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(build_variant, jobs))
    built = [r for r in results if "asm" in r]
    for r in results:
        if "error" in r:
            print(f"variant {r['i']}: not built - {r['error'][:120]}", file=sys.stderr)
    if not built:
        sys.exit("no variant built")
    os.makedirs(args.out, exist_ok=True)
    ctype = "double" if "FP64" == dtype else "float"
    # a file a kernel: their local labels are alike
    offsets = [int(o) for o in args.align.split(",") if o.strip()]
    sources, calls, wrappers, helpers = [], [], [], []
    for r in built:
        for o in offsets:
            # a copy a placement: its own symbol, its own object, o bytes past
            # a 64-byte boundary
            sym = f"{r['symbol']}_a{o}"
            sources.append(os.path.join(args.out, f"kernel{r['i']}_a{o}.s"))
            with open(sources[-1], "w") as f:
                f.write(f"\t.text\n\t.balign 64\n\t.space {o}\n" + r["asm"].replace(r["symbol"], sym))
            wrappers.append(r["wrapper"].replace(r["symbol"], sym))
            helpers.append(r["helpers"].replace(r["symbol"], sym))
            calls.append(f'    {{ pid_t p = fork(); if (0 == p) {{ OPEN_COUNTER(); '
                         f'dim_t mr = {r["mr"]}, nr = {r["nr"]}; for (int j = 0; j < nk; ++j) '
                         f'printf("variant {r["i"]} {o} %d %.2f\\n", ks[j], best({sym}, mr, nr, ks[j], a, b, c, '
                         f'{args.reps})); fflush(stdout); _exit(0); }} int st = 0; waitpid(p, &st, 0); '
                         f'if (!WIFEXITED(st) || WEXITSTATUS(st)) {{ printf("crashed {r["i"]} {o} %d\\n", '
                         f'WIFSIGNALED(st) ? WTERMSIG(st) : WEXITSTATUS(st)); fflush(stdout); }} }}')
    helpers = "".join(dict.fromkeys(helpers))
    stub = blis_stub()
    driver = (f"#define WT {ctype}\n" + stub + helpers
              + "".join(wrappers) + DRIVER.replace("VARIANTS", "\n".join(calls)))
    with open(os.path.join(args.out, "driver.c"), "w") as f:
        f.write(driver)
    with open(os.path.join(args.out, "variants.yaml"), "w") as f:
        yaml.safe_dump({r["i"]: r["settings"] for r in built}, f, sort_keys=False)
    exe = os.path.join(args.out, "bench")
    if args.perf_lib:
        # performance_counters (C++) behind three C functions
        shim = os.path.join(args.out, "pc_shim.cpp")
        with open(shim, "w") as f:
            f.write('#include "performance_counters.hpp"\n'
                    'static performance_counters* pc;\n'
                    'extern "C" int pc_init(const char* e) { pc = new performance_counters('
                    'std::vector<std::string>{e}, false); return 1; }\n'
                    'extern "C" void pc_tic() { pc->tic(); }\n'
                    'extern "C" unsigned long long pc_toc() { return pc->toc()[0]; }\n')
        steps = [[args.cc, *args.cflags.split(), "-DUSE_PCLIB", f'-DPC_EVENT="{args.event}"', "-c",
                  os.path.join(args.out, "driver.c"), "-o", os.path.join(args.out, "driver.o")],
                 [args.cxx, *args.cflags.split(), "-std=c++17", f"-I{args.perf_lib}", "-o", exe,
                  os.path.join(args.out, "driver.o"), shim,
                  os.path.join(args.perf_lib, f"performance_counters_{args.perf_backend}.cpp"), *sources]
                 + (["-lpapi"] if "papi" == args.perf_backend else [])]
    else:
        steps = [[args.cc, *args.cflags.split(), "-o", exe, os.path.join(args.out, "driver.c"), *sources]]
    for step in steps:
        r = subprocess.run(step, capture_output=True, text=True)
        if r.returncode:
            break
    if r.returncode:
        sys.exit(r.stderr[:2000])
    ks = [k for k in args.k.split(",") if k]
    if args.build_only:
        print(f"built {exe}: run '{exe} {' '.join(ks)}' on the target; variants in {args.out}/variants.yaml")
        return
    if args.remote:
        host, _, where = args.remote.partition(":")
        where = where or "."
        r = subprocess.run(["scp", exe, f"{host}:{where}/ukrsweep_bench"], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"scp failed: {r.stderr}")
        r = subprocess.run(["ssh", host, f"{where}/ukrsweep_bench {' '.join(ks)}"],
                           capture_output=True, text=True, timeout=3600)
        if r.returncode:
            print(f"ssh run ended with status {r.returncode}: {r.stderr.strip() or '(nothing on stderr)'}"
                  " - the results it printed follow", file=sys.stderr)
        out = r.stdout
    else:
        out = subprocess.run([exe, *ks], capture_output=True, text=True, timeout=3600).stdout
    for i, o, sig in re.findall(r"crashed (\d+) (\d+) (-?\d+)", out):
        print(f"variant {i} at offset {o}: crashed (signal or status {sig})", file=sys.stderr)
    found = re.search(r"# unit (\w+)", out)
    if not found:
        sys.exit("the benchmark printed nothing")
    unit = found.group(1)
    points = {}
    for i, o, k, t in re.findall(r"variant (\d+) (\d+) (\d+) ([-\d.]+)", out):
        points.setdefault((int(i), int(o)), []).append((int(k), float(t)))
    fits = {}
    for (i, o), pts in points.items():
        # least squares: t = call + step * k; the worst residual says whether
        # the points lie on a line (panels out of L1 bend it)
        nk = len(pts)
        mk, mt = sum(k for k, _ in pts) / nk, sum(t for _, t in pts) / nk
        step = sum((k - mk) * (t - mt) for k, t in pts) / max(1e-9, sum((k - mk) ** 2 for k, _ in pts))
        call = mt - step * mk
        worst = max(abs(t - (call + step * k)) / max(t, 1e-9) for k, t in pts)
        fits.setdefault(i, []).append((step, call, worst, o))
    rows = []
    for i, f in fits.items():
        f.sort()
        rows.append((f[0][0], f[-1][0], f[0][1], max(w for _, _, w, _ in f), f[0][3], i))
    rows.sort()
    settings = {r["i"]: r["settings"] for r in built}
    many = len(offsets) > 1
    print(f"{unit} per k step" + (" (best at offset .. worst)" if many else "")
          + " | per call outside the loop | worst residual | settings")
    for best_, worst_, call, resid, at, i in rows:
        spread = f" @{at:<3} .. {worst_:7.2f}" if many else ""
        print(f"{best_:10.2f}{spread} | {call:10.1f} | {100 * resid:5.1f}% | {settings[i]}")
    print("best:\n" + yaml.safe_dump(settings[rows[0][5]], sort_keys=False))


if __name__ == "__main__":
    main()

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

the data type is the config's own - sgemm.yaml is FP32, dgemm.yaml FP64 -
unless --dt names it, so a file named otherwise (gemm-fp32.yaml) can be swept

any-kernel mode - packm families always, gemm with --any: the kernel called
through its own symbol, its arguments from what it says of itself (its loop's
count, its components' bases and strides from the family's layouts, packed
tightly, and its sizes) - no BLIS wrapper; timed at loop counts (--loop, a
line fitted as for k) or at working sets filling cache levels (--levels, the
loop count whose components take them: a time a loop step and bytes a cycle):

  ukropt-sweep ukrconfig/avx2/fma_bc_blocks/dpackm.yaml --family pk --case trans \\
      --size 8 --levels L1=16K,L2=512K,L3=16M,MEM=1G [--grid grid.yaml]

BLIS-like packm mode (--blis) - a packm family called as BLIS calls it, through
ukrgen's dispatch with BLIS's arguments (cdim, inca, lda, ldp - strides in
elements, as BLIS's), at numbers of columns: the case each call falls into (or the
reference, or both for what a case leaves), its panel checked against the
reference's (the run fails where one differs), its time a call and a column;
--case keeps only some kinds of case, the dispatch falling into what's left:

  ukropt-sweep ukrconfig/avx2/fma_bc_blocks/dpackm.yaml --family pk \\
      --blis cdim=4,inca=1024,lda=1,ldp=4 --loop 64,256,1001 [--case trans]

grid.yaml - family keys or flow parameters to lists of values:
  sched-cycle-slots: ["2:vl|vs|vf,sl,i", "3:vf|vs|vf,vl|vs|vf,sl,sl,i;vf-sl"]
  buckets: [null, "vf:5", "vf:8"]
  schedule: [["RAW:data::::13"], ["RAW:data:freg:::7", "RAW:data:vreg:::7"]]
"""

import argparse
import copy
import itertools
import os
import platform
import random
import re
import shlex
import statistics
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
from ukropt import anykernel, blislike                                                  # noqa: E402

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
UKRSWEEP_GEMM
typedef void (*ukr_t)(dim_t, dim_t, dim_t, const void*, const void*, const void*, const void*, void*,
                      inc_t, inc_t, const auxinfo_t*, const cntx_t*);
static double best(ukr_t f, dim_t mr, dim_t nr, dim_t k, const WT* a, const WT* b, WT* c, int reps)
{
    WT alpha = 1, beta = 1; double t = 1e300;
    for (int r = 0; r < reps; ++r) {
#ifdef USE_PCLIB
        pc_tic();
        for (int i = 0; i < ITERS; ++i) f(mr, nr, k, &alpha, a, b, &beta, c, 1, mr, NULL, NULL);
        double d = (double)pc_toc() / ITERS;
#else
        double t0 = now();
        for (int i = 0; i < ITERS; ++i) f(mr, nr, k, &alpha, a, b, &beta, c, 1, mr, NULL, NULL);
        double d = (now() - t0) / ITERS;
#endif
        if (d < t) t = d;
    }
    return t;
}
#ifndef UKRSWEEP_PROLOGUE
#define UKRSWEEP_PROLOGUE(argc, argv) ((void)0)
#endif
int main(int argc, char** argv)
{
    /* a core's preparation, first of all (--use-prologue) */
    UKRSWEEP_PROLOGUE(argc, argv);
    int nk = argc - 1, ks[64];
    for (int i = 0; i < nk && i < 64; ++i) ks[i] = atoi(argv[i + 1]);
    /* pinned where asked (UKRSWEEP_CPU): the children inherit it */
    const char *cpu = getenv("UKRSWEEP_CPU");
    if (cpu && *cpu) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(atoi(cpu), &set);
        if (sched_setaffinity(0, sizeof set, &set))
            perror("sched_setaffinity");
    }
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


# the drivers' shared head - includes, counters, the clock - and gemm's own
DRIVER_COMMON, _, _GEMM = DRIVER.partition("UKRSWEEP_GEMM\n")
DRIVER = DRIVER_COMMON + _GEMM


def load_prologue(name: str | None) -> str:
    """
    A prologue for the benchmark (--use-prologue): ukropt/prologues/NAME.h,
    or a header by path. It defines UKRSWEEP_PROLOGUE(argc, argv), which the
    benchmark's main runs first
    """
    if not name:
        return ""
    if os.sep in name or name.endswith(".h"):
        return open(name).read() + "\n"
    from importlib.resources import files
    here = files("ukropt") / "prologues"
    path = here / f"{name}.h"
    if not path.is_file():
        known = sorted(p.name[:-2] for p in here.iterdir() if p.name.endswith(".h"))
        sys.exit(f"no prologue {name!r}: there are {', '.join(known)} (or give a header's path)")
    return path.read_text() + "\n"


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


CTYPE = {"FP64": "double", "FP32": "float"}


def dtype_of(config: str) -> str:
    """
    The data type a gemm family file is for: its name's first letter, as
    ukrgen's configs are named (sgemm, spackm - FP32, else FP64)
    """
    return "FP32" if os.path.basename(config).startswith("s") else "FP64"


def build_variant(job):
    """
    One variant, in a worker process: the kernel's assembly, BLIS wrapper and
    vector-length helpers as text, or why it didn't build
    """
    i, settings, config, dtype, family_name, m, n = job
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
    ctype = CTYPE[dtype]
    # its vector length function named before the wrapper and sizes read it
    helpers = vl_helpers([k], make_generator(cfg.isa))
    return {"i": i, "settings": settings, "asm": asm_source(k), "wrapper": gemm_wrapper(k),
            "helpers": helpers, "symbol": k.symbol,
            "mr": c_expr(k.mr, ctype, k.vl_fn), "nr": c_expr(k.nr, ctype, k.vl_fn)}


def main():
    p = argparse.ArgumentParser(description=__doc__, epilog=MODES_HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="a family file (ukrconfig/<isa>/<config>/dgemm.yaml, dpackm.yaml)")
    p.add_argument("--family", required=True)
    p.add_argument("--size", default=None,
                   help="MxN in the family's units (2x14), a packm case's rows (8)")
    p.add_argument("--blis", default=None,
                   help="BLIS-like packm mode: the family called as BLIS calls it, through ukrgen's dispatch "
                        "with these arguments - cdim=8,inca=1,lda=1024,ldp=8 (cdim_max, n_extra, kappa; "
                        "strides in elements as BLIS's, lda cdim where inca is 1 and lda isn't given) - at "
                        "numbers of columns --loop: the case it falls into, its panel checked against the "
                        "reference's, its time")
    p.add_argument("--grid", default=None,
                   help="YAML: family keys or flow parameters to lists of values (none: the family's own)")
    p.add_argument("--kind", default=None, choices=["gemm", "packm"],
                   help="the family file's kind (default: its name - dpackm.yaml packm, else gemm)")
    p.add_argument("--case", default=None,
                   help="a packm family's case: vec, gather, trans (default: its first); with --blis, the "
                        "cases kept by name or kind, vec,trans - the dispatch falls into what's left (a grid's "
                        "dispatch: [trans, gather] orders them)")
    p.add_argument("--any", action="store_true",
                   help="any-kernel mode for gemm too (packm always): the kernel called through its own "
                        "symbol, its arguments from its layouts and sizes - no BLIS wrapper")
    p.add_argument("--loop", default=None,
                   help="any-kernel mode: the loop's counts to time, a line fitted (default: --k's)")
    p.add_argument("--levels", default=None,
                   help="any-kernel mode: working sets to fill, NAME=BYTES - L1=32K,L2=1M,L3=16M,MEM=1G - "
                        "each timed at the loop count whose components take them, its rate in bytes")
    p.add_argument("--sample", type=int, default=None, help="a random sample of the combinations")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cc", default="gcc")
    p.add_argument("--cflags", default="-O2")
    p.add_argument("--k", default="16,32,48,64,80",
                   help="values of k to time, a line fitted through them - keep A's and B's panels "
                        "((MR+NR) x k elements) in L1")
    p.add_argument("-j", "--jobs", type=int, default=os.cpu_count(), help="variants built in parallel")
    p.add_argument("--cpu", default=None,
                   help="the core(s) to measure on, 1 or 1,5 (the benchmark pins itself; on the "
                        "K1 the clusters differ by a few cycles a k step)")
    p.add_argument("--rank", default=None,
                   help="with several cores, rank by: mean (default), max (the worst core), min, "
                        "or a core's number")
    p.add_argument("--use-prologue", default=None,
                   help="prepare the core first: a name in ukropt/prologues (k3_ai: the K3's AI "
                        "cores) or a header's path defining UKRSWEEP_PROLOGUE(argc, argv)")
    p.add_argument("--runs", "--repeat", dest="repeat", type=int, default=1,
                   help="the whole benchmark run this many times, each point the median of "
                        "the runs' fastest")
    p.add_argument("--max-residual", type=float, default=2.0,
                   help="fits with a worse residual (percent) are marked ? and not chosen best")
    p.add_argument("--align", default="0",
                   help="byte offsets from a 64-byte boundary to place each kernel at (multiples of 4: "
                        "0,4,8,16,32): each variant measured at each, best and worst reported - how much "
                        "of a difference is placement")
    p.add_argument("--measurements", "--reps", dest="reps", type=int, default=20,
                   help="timed measurements per variant and k in a run, the fastest kept")
    p.add_argument("--iterations", type=int, default=64,
                   help="kernel calls back to back in one measurement")
    p.add_argument("--out", default="ukrsweep")
    p.add_argument("--build-only", action="store_true", help="build the binary, run it elsewhere")
    p.add_argument("--runner", default=None,
                   help="a command to run the binary with here - 'qemu-riscv64-static -cpu "
                        "rv64,v=true,vlen=256': which case a call falls into and whether its panel is "
                        "right, checked before measuring on the target (the times emulated)")
    p.add_argument("--remote", default=None,
                   help="HOST:DIR - copy the binary there (scp) and run it (ssh): generate and "
                        "cross-compile here, measure on the target")
    p.add_argument("--perf-lib", default=None,
                   help="a clone of github.com/linedot/performance_counters: its counters instead")
    p.add_argument("--perf-backend", default="perf", help="perf, papi, m1 or simple")
    p.add_argument("--event", default="CYCLES", help="the counter (PAPI: PAPI_TOT_CYC)")
    p.add_argument("--cxx", default="g++")
    p.add_argument("--dt", "--dtype", dest="dt", default=None, choices=sorted(CTYPE),
                   help="the data type the kernel is for: FP64, FP32 (default: the family file's "
                        "name - sgemm.yaml is FP32, dgemm.yaml FP64)")
    args = p.parse_args()

    kind = args.kind or ("packm" if "packm" in os.path.basename(args.config) else "gemm")
    mode = "blis" if args.blis else ("any" if args.any or "gemm" != kind else "gemm")
    check_options(p, args, mode, kind)
    if args.blis:
        return blis_mode(args, kind)
    if not args.size:
        sys.exit("--size: MxN for gemm, a packm case's rows")
    anyk = args.any or "gemm" != kind
    dtype = args.dt or dtype_of(args.config)
    cfg = configuration.from_files({(kind, dtype): args.config})
    check_toolchain(args, cfg.isa)
    base = next(f for f in cfg.families if f.name == args.family)
    size = tuple(int(x) for x in args.size.lower().split("x"))
    if "gemm" == kind and 2 != len(size):
        sys.exit("--size MxN for a gemm family")
    grid = yaml.safe_load(open(args.grid)) if args.grid else {}
    variants = combinations(grid, args.sample, args.seed) if grid else [{}]
    if anyk:
        case = args.case or (base.cases[0][0] if "packm" == kind else None)
        jobs = [(i, v, args.config, kind, dtype, args.family, case, size) for i, v in enumerate(variants)]
        worker = anykernel.build_any_variant
    else:
        m, n = size
        jobs = [(i, v, args.config, dtype, args.family, m, n) for i, v in enumerate(variants)]
        worker = build_variant
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(worker, jobs))
    built = [r for r in results if "asm" in r]
    for r in results:
        if "error" in r:
            print(f"variant {r['i']}: not built - {r['error'][:120]}", file=sys.stderr)
    if not built:
        sys.exit("no variant built")
    os.makedirs(args.out, exist_ok=True)
    ctype = CTYPE[dtype]
    # a file a kernel: their local labels are alike
    offsets = [int(o) for o in args.align.split(",") if o.strip()]
    gen = make_generator(cfg.isa)
    sources, calls, wrappers, helpers = [], [], [], []
    for r in ([] if anyk else built):
        for o in offsets:
            # a copy a placement: its own symbol, its own object, o bytes past
            # a 64-byte boundary
            sym = f"{r['symbol']}_a{o}"
            sources.append(os.path.join(args.out, f"kernel{r['i']}_a{o}.s"))
            with open(sources[-1], "w") as f:
                # nops, not .space's zeros (asmgen's align)
                f.write(gen.asmwrap(".text") + gen.align(64, o) + r["asm"].replace(r["symbol"], sym))
            wrappers.append(r["wrapper"].replace(r["symbol"], sym))
            helpers.append(r["helpers"].replace(r["symbol"], sym))
            calls.append(f'    {{ pid_t p = fork(); if (0 == p) {{ OPEN_COUNTER(); '
                         f'dim_t mr = {r["mr"]}, nr = {r["nr"]}; for (int j = 0; j < nk; ++j) '
                         f'printf("variant {r["i"]} {o} %d %.2f\\n", ks[j], best({sym}, mr, nr, ks[j], a, b, c, '
                         f'{args.reps})); fflush(stdout); _exit(0); }} int st = 0; waitpid(p, &st, 0); '
                         f'if (!WIFEXITED(st) || WEXITSTATUS(st)) {{ printf("crashed {r["i"]} {o} %d\\n", '
                         f'WIFSIGNALED(st) ? WTERMSIG(st) : WEXITSTATUS(st)); fflush(stdout); }} }}')
    stub = blis_stub()
    prefix = ("#define _GNU_SOURCE\n#include <sched.h>\n#include <stdlib.h>\n"
              + load_prologue(args.use_prologue) + f"#define WT {ctype}\n#define ITERS {args.iterations}\n" + stub)
    if anyk:
        # each kernel called through its own symbol, its arguments from its
        # layouts and sizes
        head, variants_c, placed = anykernel.any_driver(built, dtype, offsets, args.reps)
        for r, o, sym in placed:
            sources.append(os.path.join(args.out, f"kernel{r['i']}_a{o}.s"))
            with open(sources[-1], "w") as f:
                f.write(gen.asmwrap(".text") + gen.align(64, o) + r["asm"].replace(r["symbol"], sym))
            helpers.append(r["helpers"].replace(r["symbol"], sym))
        driver = (prefix + "".join(dict.fromkeys(helpers)) + "static WT one = 1;\n" + DRIVER_COMMON + head
                  + anykernel.ANY_DRIVER.replace("VARIANTS", variants_c))
    else:
        helpers = "".join(dict.fromkeys(helpers))
        driver = prefix + helpers + "".join(wrappers) + DRIVER.replace("VARIANTS", "\n".join(calls))
    with open(os.path.join(args.out, "driver.c"), "w") as f:
        f.write(driver)
    with open(os.path.join(args.out, "variants.yaml"), "w") as f:
        yaml.safe_dump({r["i"]: r["settings"] for r in built}, f, sort_keys=False)
    exe = compile_bench(args, sources)
    ks = [k for k in ((args.levels or args.loop or args.k) if anyk else args.k).split(",") if k]
    if args.build_only:
        print(f"built {exe}: run '{exe} {' '.join(ks)}' on the target; variants in {args.out}/variants.yaml")
        return
    if args.remote:
        host, _, where = args.remote.partition(":")
        where = where or "."
        r = subprocess.run(["scp", exe, f"{host}:{where}/ukrsweep_bench"], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"scp failed: {r.stderr}")
    cpus = [int(c) for c in str(args.cpu).split(",") if c.strip()] if args.cpu is not None else [None]
    rank = args.rank or ("mean" if len(cpus) > 1 else "min")
    if rank not in ("mean", "max", "min") and not (rank.isdigit() and int(rank) in cpus):
        sys.exit(f"--rank {rank}: mean, max, min or one of the cores measured ({args.cpu})")
    runs = {}
    for cpu in cpus:
        pin = f"UKRSWEEP_CPU={cpu} " if cpu is not None else ""
        env = dict(os.environ, **({"UKRSWEEP_CPU": str(cpu)} if cpu is not None else {}))
        outs = []
        for _ in range(max(1, args.repeat)):
            if args.remote:
                r = subprocess.run(["ssh", host, f"cd {where} && {pin}./ukrsweep_bench {' '.join(ks)}"],
                                   capture_output=True, text=True, timeout=3600)
                if r.returncode:
                    print(f"ssh run (core {cpu}) ended with status {r.returncode}: "
                          f"{r.stderr.strip() or '(nothing on stderr)'} - the results it printed follow",
                          file=sys.stderr)
                outs.append(r.stdout)
            else:
                outs.append(subprocess.run([*shlex.split(args.runner or ""), exe, *ks], capture_output=True,
                                           text=True, timeout=3600,
                                           env=env).stdout)
        runs[cpu] = "\n".join(outs)
    out = "\n".join(runs.values())
    for i, o, sig in sorted(set(re.findall(r"crashed (\d+) (\d+) (-?\d+)", out))):
        print(f"variant {i} at offset {o}: crashed (signal or status {sig})", file=sys.stderr)
    found = re.search(r"# unit (\w+)", out)
    if not found:
        sys.exit("the benchmark printed nothing")
    unit = found.group(1)
    if anyk and args.levels:
        anykernel.report_levels(runs, {r["i"]: r["settings"] for r in built}, unit)
        return
    # per core: each point the median of its runs, a line per variant and
    # placement - least squares, t = call + step * k, its worst residual -
    # and per variant its best placement: (step, worst step, call, worst
    # residual, offset)
    per_cpu = {}
    for cpu, text in runs.items():
        raw = {}
        for i, o, k, t in re.findall(r"variant (\d+) (\d+) (\d+) ([-\d.]+)", text):
            raw.setdefault((int(i), int(o)), {}).setdefault(int(k), []).append(float(t))
        fits = {}
        for (i, o), per in raw.items():
            pts = [(k, statistics.median(ts)) for k, ts in sorted(per.items())]
            nk = len(pts)
            mk, mt = sum(k for k, _ in pts) / nk, sum(t for _, t in pts) / nk
            step = sum((k - mk) * (t - mt) for k, t in pts) / max(1e-9, sum((k - mk) ** 2 for k, _ in pts))
            call = mt - step * mk
            worst = max(abs(t - (call + step * k)) / max(t, 1e-9) for k, t in pts)
            fits.setdefault(i, []).append((step, call, worst, o))
        per_cpu[cpu] = {i: (min(f)[0], max(f)[0], min(f)[1], max(w for _, _, w, _ in f), min(f)[3])
                        for i, f in fits.items()}
    variants = sorted(set.intersection(*(set(v) for v in per_cpu.values())))
    if not variants:
        sys.exit("no variant measured on every core")

    def score(i):
        steps = [per_cpu[c][i][0] for c in cpus]
        if "mean" == rank:
            return sum(steps) / len(steps)
        if "max" == rank:
            return max(steps)
        if "min" == rank:
            return min(steps)
        return per_cpu[int(rank)][i][0]

    settings = {r["i"]: r["settings"] for r in built}
    limit = args.max_residual / 100
    resid = {i: max(per_cpu[c][i][3] for c in cpus) for i in variants}
    rows = sorted(variants, key=score)
    many, multi = len(offsets) > 1, len(cpus) > 1
    if multi:
        cols = " | ".join(f"core {c}" + (" @best offset" if many else "") for c in cpus)
        print(f"{unit} per {'loop' if anyk else 'k'} step, {rank} over cores | {cols} | per call outside the loop | "
              "worst residual | settings")
    else:
        print(f"{unit} per {'loop' if anyk else 'k'} step" + (" (best at offset .. worst)" if many else "")
              + " | per call outside the loop | worst residual | settings")
    for i in rows:
        flag = "?" if resid[i] > limit else " "
        if multi:
            cols = " | ".join(f"{per_cpu[c][i][0]:7.2f}" + (f" @{per_cpu[c][i][4]:<3}" if many else "")
                              for c in cpus)
            call = sum(per_cpu[c][i][2] for c in cpus) / len(cpus)
            print(f"{score(i):10.2f} | {cols} | {call:10.1f} | {100 * resid[i]:5.1f}%{flag}| {settings[i]}")
        else:
            best_, worst_, call, _, at = per_cpu[cpus[0]][i]
            spread = f" @{at:<3} .. {worst_:7.2f}" if many else ""
            print(f"{best_:10.2f}{spread} | {call:10.1f} | {100 * resid[i]:5.1f}%{flag}| {settings[i]}")
    if any(resid[i] > limit for i in variants):
        print(f"? a residual above {args.max_residual}%: the points aren't a line - not chosen best "
              "(--runs, --cpu)")
    trusted = [i for i in rows if resid[i] <= limit] or rows
    print("best:\n" + yaml.safe_dump(settings[trusted[0]], sort_keys=False))


#: the machine each ISA's kernels run on, as compilers name it (-dumpmachine)
ISA_MACHINE = {"rvv": "riscv64", "neon": "aarch64", "sve": "aarch64", "sme": "aarch64", "avx": "x86_64"}
#: a compiler and flags for each machine, and a way to run it here
CROSS = {"riscv64": ("riscv64-linux-gnu-gcc", "-O2 -static -march=rv64gcv",
                     "qemu-riscv64-static -cpu rv64,v=true,vlen=256,elen=64"),
         "aarch64": ("aarch64-linux-gnu-gcc", "-O2 -static -march=armv9-a+sve2+sme", "qemu-aarch64-static -cpu max"),
         "x86_64": ("gcc", "-O2 -march=native", None)}


def check_toolchain(args, isa: str):
    """
    The compiler builds for the machine the config's kernels are for, and
    the binary runs where it's run - else the sweep stops, saying what to
    give (a cross compiler's driver.c gets the kernels' inline assembly - an
    x86 compiler fills RVV's vsetvli %0 with %rcx)
    """
    machine = next((m for prefix, m in ISA_MACHINE.items() if isa.startswith(prefix)), None)
    if machine is None:
        return
    try:
        target = subprocess.run([args.cc, "-dumpmachine"], capture_output=True, text=True,
                                timeout=60).stdout.strip()
    except OSError:
        sys.exit(f"--cc {args.cc}: no such compiler")
    cc, cflags, runner = CROSS[machine]
    if target and not target.startswith(machine):
        sys.exit(f"{os.path.basename(args.config)} is for {isa}, a {machine} ISA; --cc {args.cc} compiles for "
                 f"{target}: give --cc {cc} --cflags '{cflags}'")
    host = platform.machine()
    if machine != host and not (args.remote or args.runner or args.build_only):
        sys.exit(f"a {machine} binary doesn't run on this {host}: --remote HOST:DIR measures it on the target"
                 + (f", --runner '{runner}' checks it here (its times emulated)" if runner else "")
                 + ", --build-only builds it")


def compile_bench(args, sources: list) -> str:
    """
    The benchmark from driver.c and the kernels' sources - with
    performance_counters' counters where --perf-lib gives them - or the
    compiler's complaint
    """
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
            sys.exit(r.stderr[:2000])
    return exe


#: what each option is in each mode (gemm, any-kernel, BLIS-like): None where
#: the mode doesn't use it - given there, the sweep stops and says why
OPTIONS = {
    "size": ("--size", {"gemm": "MxN", "any": "MxN (gemm), a packm case's rows",
                        "blis": None}, "cdim is --blis's"),
    "case": ("--case", {"gemm": None, "any": "the packm case (default the family's first)",
                        "blis": "the case kinds kept: vec,trans"}, "a gemm family has no cases"),
    "any": ("--any", {"gemm": "any-kernel mode instead", "any": "(a packm family's mode anyway)",
                      "blis": None}, "--blis calls the family's dispatch, not a kernel"),
    "k": ("--k", {"gemm": "the values of k", "any": None, "blis": None},
          "--loop gives the counts (its default --k's)"),
    "loop": ("--loop", {"gemm": None, "any": "loop counts, a line fitted (default --k's)",
                        "blis": "numbers of columns (default --k's)"}, "gemm mode times --k's values"),
    "levels": ("--levels", {"gemm": None, "any": "working sets NAME=BYTES, a rate each - not with --loop",
                            "blis": None}, "any-kernel mode's (packm, or gemm with --any)"),
    "align": ("--align", {"gemm": "offsets", "any": "offsets", "blis": None},
              "--blis places each kernel once"),
    "rank": ("--rank", {"gemm": "with several --cpu", "any": "with several --cpu, --loop", "blis": None},
             "nothing's ranked: each variant reported"),
    "max_residual": ("--max-residual", {"gemm": "the fit's", "any": "--loop's fit", "blis": None},
                     "no line is fitted"),
}

MODES_HELP = """
modes - the family file's kind (its name, or --kind) and two options choose one:
  gemm        a gemm family (dgemm.yaml): each variant through its BLIS wrapper at
              the values of --k, a line fitted - cycles a k step, a call's
  any-kernel  a packm family (dpackm.yaml), or a gemm family with --any: each
              variant through its own symbol, at the loop counts of --loop (a line
              fitted) or at the working sets of --levels (a rate per level)
  BLIS-like   a packm family with --blis: its dispatch called with BLIS's
              arguments at the numbers of columns of --loop - the case fallen
              into, the panel checked, the time

options by mode - gemm | any-kernel | BLIS-like; where a mode has -, the option
given stops the sweep, saying why:
  --size          MxN | MxN (gemm), a packm case's rows | - (cdim is --blis's)
  --case          - | the packm case (default the first) | the case kinds kept
  --any           any-kernel mode instead | (packm: anyway) | -
  --k             the values of k | - (--loop's default) | - (--loop's default)
  --loop          - | loop counts, a line fitted | numbers of columns
  --levels        - | working sets, a rate each; not with --loop | -
  --align         offsets | offsets | - (one placement each)
  --cpu           cores | cores | one core
  --rank          with several cores | with several cores, --loop | -
  --max-residual  the fit's | --loop's fit | -
  --sample/--seed with --grid; --perf-backend/--event/--cxx with --perf-lib
  every other option in every mode: --grid --dt --kind --cc --cflags --out
  --build-only --remote --jobs --measurements --iterations --runs
  --use-prologue --perf-lib
the defaults from the file's name: its kind (dpackm.yaml packm, else gemm) and
its type (sgemm.yaml, spackm.yaml FP32, else FP64), --dt and --kind over them
"""


def check_options(p, args, mode: str, kind: str):
    """
    Each option given (not at its default) used by the mode, and the ones
    that go together together: else the sweep stops, saying why
    """
    given = lambda dest: getattr(args, dest) != p.get_default(dest)
    for dest, (flag, by_mode, why) in OPTIONS.items():
        if given(dest) and by_mode[mode] is None:
            sys.exit(f"{flag}: not in {mode} mode - {why}")
    if "any" == mode and "gemm" == kind and given("case"):
        sys.exit("--case: a gemm family has no cases")
    if given("levels") and given("loop"):
        sys.exit("--levels or --loop: the counts for working sets, or the counts themselves")
    if given("levels") and (given("rank") or given("max_residual")):
        sys.exit("--rank, --max-residual: with --levels no line is fitted, nothing ranked")
    if "blis" == mode and args.cpu is not None and "," in str(args.cpu):
        sys.exit("--cpu: --blis measures on one core")
    if not given("grid") and (given("sample") or given("seed")):
        sys.exit("--sample, --seed: they choose among --grid's combinations")
    if not given("perf_lib") and any(given(d) for d in ("perf_backend", "event", "cxx")):
        sys.exit("--perf-backend, --event, --cxx: with --perf-lib")


def blis_mode(args, kind: str):
    """
    BLIS-like packm mode (--blis): every variant's whole family and its
    dispatch in one benchmark, called with BLIS's arguments as given at each
    number of columns - the case each falls into, the reference's calls,
    whether its panel is the reference's, its time
    """
    if "packm" != kind:
        sys.exit("--blis: a packm family file")
    ns = [n for n in (args.loop or args.k).split(",") if n]
    call = blislike.resolve(blislike.parse_call(args.blis), [int(n) for n in ns])
    dtype = args.dt or dtype_of(args.config)
    cfg = configuration.from_files({(kind, dtype): args.config})
    check_toolchain(args, cfg.isa)
    grid = yaml.safe_load(open(args.grid)) if args.grid else {}
    variants = combinations(grid, args.sample, args.seed) if grid else [{}]
    # --case: the cases kept by name or kind (vec,trans) - the dispatch falls
    # into what's left; a call none of them can take refused (its numbers
    # would be the reference's), one no case can take said
    keep = {c.strip() for c in (args.case or "").split(",") if c.strip()}
    base = next(f for f in cfg.families if f.name == args.family)
    kinds = {name: kind for name, kind in base.case_kinds.items()
             if not keep or name in keep or kind in keep}
    if keep and not kinds:
        sys.exit(f"--case: no case of {args.family} is named or of a kind {sorted(keep)}: "
                 f"{sorted(base.case_kinds)}")
    if not blislike.reachable(call, kinds):
        why = "; ".join(f"{name} ({kind}) needs {blislike.NEEDS[kind][0]}" for name, kind in kinds.items()
                        if kind in blislike.NEEDS)
        if keep:
            sys.exit(f"--case: the call (inca {call['inca']}, lda {call['lda']}, cdim {call['cdim']} of "
                     f"{call['cdim_max']}, ldp {call['ldp']}) falls into none of the cases kept - {why}")
        print(f"note: no case packs this call - the reference's numbers ({why})", file=sys.stderr)
    jobs = [(i, v, args.config, dtype, args.family, keep) for i, v in enumerate(variants)]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(blislike.build_blis_variant, jobs))
    built = [r for r in results if "dispatch" in r]
    for r in results:
        if "error" in r:
            print(f"variant {r['i']}: not built - {r['error'][:120]}", file=sys.stderr)
    # a kernel a variant didn't build said - a call falling into the
    # reference for it is no case's measurement - and a variant none of
    # whose kept cases built refused, not measured as the reference
    kept = []
    for r in built:
        for line in r["skipped"]:
            print(f"variant {r['i']} {r['settings']}: not built {line}", file=sys.stderr)
        if keep and not any(name in keep or kind in keep for name, kind in r["kept_as"]):
            print(f"variant {r['i']} {r['settings']}: none of the cases kept ({', '.join(sorted(keep))}) "
                  f"built - not measured", file=sys.stderr)
            continue
        kept.append(r)
    built = kept
    if not built:
        sys.exit("no variant built")
    os.makedirs(args.out, exist_ok=True)
    gen = make_generator(cfg.isa)
    sources = []
    for r in built:
        for j, (sym, asm) in enumerate(r["asm"]):
            sources.append(os.path.join(args.out, f"variant{r['i']}_{j}.s"))
            with open(sources[-1], "w") as f:
                f.write(gen.asmwrap(".text") + gen.align(64, 0) + asm)
    head, variants_c = blislike.blis_driver(built, call, args.reps)
    driver = ("#define _GNU_SOURCE\n#include <sched.h>\n#include <stdlib.h>\n" + load_prologue(args.use_prologue)
              + f"#define WT {CTYPE[dtype]}\n#define ITERS {args.iterations}\n" + blis_stub()
              + "".join(dict.fromkeys(r["helpers"] for r in built)) + DRIVER_COMMON + head
              + blislike.BLIS_DRIVER.replace("VARIANTS", variants_c))
    with open(os.path.join(args.out, "driver.c"), "w") as f:
        f.write(driver)
    with open(os.path.join(args.out, "variants.yaml"), "w") as f:
        yaml.safe_dump({r["i"]: r["settings"] for r in built}, f, sort_keys=False)
    exe = compile_bench(args, sources)
    if args.build_only:
        print(f"built {exe}: run '{exe} {' '.join(ns)}' on the target; variants in {args.out}/variants.yaml")
        return
    if args.remote:
        host, _, where = args.remote.partition(":")
        where = where or "."
        r = subprocess.run(["scp", exe, f"{host}:{where}/ukrsweep_bench"], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"scp failed: {r.stderr}")
    pin = f"UKRSWEEP_CPU={args.cpu} " if args.cpu is not None else ""
    env = dict(os.environ, **({"UKRSWEEP_CPU": str(args.cpu)} if args.cpu is not None else {}))
    outs = []
    for _ in range(max(1, args.repeat)):
        # each point the median of the runs' fastest (--runs)
        if args.remote:
            outs.append(subprocess.run(["ssh", host, f"cd {where} && {pin}./ukrsweep_bench {' '.join(ns)}"],
                                       capture_output=True, text=True, timeout=3600).stdout)
        else:
            outs.append(subprocess.run([*shlex.split(args.runner or ""), exe, *ns], capture_output=True,
                                       text=True, timeout=3600, env=env).stdout)
    out = "\n".join(outs)
    for i, _, sig in sorted(set(re.findall(r"crashed (\d+) (\d+) (-?\d+)", out))):
        print(f"variant {i}: crashed (signal or status {sig})", file=sys.stderr)
    found = re.search(r"# unit (\w+)", out)
    if not found:
        sys.exit("the benchmark printed nothing")
    esize = 8 if "FP64" == dtype else 4
    if not blislike.report_blis({None: out}, built, call, found.group(1), esize):
        sys.exit(1)


if __name__ == "__main__":
    main()

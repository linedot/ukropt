"""
ukropt-fmaload: FMA throughput with vector loads among the FMAs - FLOP per
cycle at FMA-to-load ratios, for data types and FMA forms. The measurement
behind the FMA throughput table of our paper, built on ukropt-pairs.

    ukropt-fmaload rvv --ratios 0,1,4,14,26 --dt FP64,FP32,FP16 --form vf,vv ...

A loop's body updates N accumulators, each once per N FMAs, from one vector
source and one scalar (vf) or vector (vv) operand, with a vector load after
every R-th FMA - counted across the loop's edge too: the body is lcm(N, R)
FMAs long, so the loads are exactly R FMAs apart everywhere (ratio 0: FMAs
only, the baseline). The loads go round-robin into --load-regs registers no
FMA reads, from a 4096-aligned buffer holding the data type's 1.0. FLOP per
cycle uses the vector length the core reports at run time.
"""

import argparse
import math
import os
import re
import sys

from .harness import C_HEADER, C_MAIN_BEGIN, C_MAIN_END, add_common_args, build, load_prologue, run
from .pairs import BUF_BYTES, isa_ops, isa_params, parse

EBYTES = {"FP64": 8, "FP32": 4, "FP16": 2}
MAX_BODY = 1024        # FMAs a loop's body may hold

MEASURE = r"""
typedef void (*fl_fn)(long, double*);
static double buf[BUF_DOUBLES] __attribute__((aligned(4096)));
static int by_value(const void *a, const void *b)
{
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}
/* per iteration: the least and the median of reps timings of n iterations -
   n large enough that the call's cost doesn't count, no difference of two
   lengths (whose minima can come from different modes of the core) */
static void measure(const char *name, fl_fn f, long n, int reps)
{
    IN_CHILD(name, {
        double t[64];
        if (reps > 64) reps = 64;
        if (reps < 1) reps = 1;
        f(n / 4 + 1, buf);
        for (int r = 0; r < reps; ++r) {
            double t0 = now(); TIC();
            f(n, buf);
            t[r] = TOC(t0);
        }
        qsort(t, reps, sizeof *t, by_value);
        printf("pair %s %.4f\n", name, t[0] / n);
        printf("median %s %.4f\n", name, t[reps / 2] / n);
    });
}
"""

FILL = r"""
/* iterations for a body of fmas FMAs: n, or enough for min_fmas FMAs */
static long min_fmas = 1000000;
static long iters(long n, long fmas)
{
    long need = (min_fmas + fmas - 1) / fmas;
    return n > need ? n : need;
}
/* the buffer full of the data type's 1.0 */
static void fill(int bytes)
{
    unsigned char *p = (unsigned char*)buf;
    for (size_t i = 0; i + bytes <= sizeof buf; i += bytes) {
        if (8 == bytes) { double v = 1.0; memcpy(p + i, &v, 8); }
        else if (4 == bytes) { float v = 1.0f; memcpy(p + i, &v, 4); }
        else { uint16_t v = 0x3C00; memcpy(p + i, &v, 2); }
    }
}
"""


def free_vregs(ops) -> int:
    """
    The vector registers an experiment can have on this ISA
    """
    rt = ops.gen.get_reg_tracker()
    n = 0
    while n < 1024:
        try:
            rt.reserve_any_reg("vreg")
        except Exception:  # pylint: disable=broad-except
            break
        n += 1
    return n


def body_for(form: str, n_acc: int, ratio: int, load_regs: int) -> tuple[list, int, int]:
    """
    A loop's body: N accumulators' FMAs, a load after every R-th, lcm(N, R)
    FMAs long. The body, its FMAs, its loads
    """
    length = n_acc if not ratio else math.lcm(n_acc, ratio)
    out, loads = [], 0
    for i in range(length):
        if "vf" == form:
            out.append(["fma", f"c{i % n_acc}", "a0", "s0"])
        else:
            out.append(["vfma", f"c{i % n_acc}", "a0", "b0"])
        if ratio and (i + 1) % ratio == 0:
            out.append(["vload", f"v{loads % load_regs}"])
            loads += 1
    return out, length, loads


def fp16_march(cflags: str) -> str:
    """
    RISC-V's FP16: zfh (scalars) and zvfh (vectors) added to -march where
    missing
    """
    def add(m):
        march = m.group(0)
        for ext in ("zfh", "zvfh"):
            if not re.search(rf"_{ext}(?![a-z])", march):
                march += f"_{ext}"
        return march
    return re.sub(r"-march=rv(32|64)[a-z0-9_]*", add, cflags)


def main():
    p = argparse.ArgumentParser(
        prog="ukropt-fmaload",
        description="FMA throughput with vector loads every R FMAs: FLOP/cycle by ratio, data "
                    "type and FMA form - loads exactly R FMAs apart, across the loop's edge too")
    p.add_argument("isa", help="an ISA as ukrgen names it: rvv, neon, sve, avx256, avx512, ...")
    p.add_argument("--ratios", default="0,1,4,14,26",
                   help="FMAs per vector load, comma-separated; 0 is the baseline, FMAs only")
    p.add_argument("--dt", default="FP64", help="data types, comma-separated: FP64,FP32,FP16")
    p.add_argument("--form", default="vf",
                   help="FMA forms, comma-separated: vf (a scalar operand, as GEMM kernels on "
                        "RVV), vv (vector by vector)")
    p.add_argument("--isa-param", action="append", default=[], metavar="NAME=VALUE",
                   help="a generator parameter (RVV: LMUL=2), repeatable; RVV's ZVFH is on for "
                        "FP16 by itself")
    p.add_argument("--accumulators", type=int, default=None,
                   help="independent accumulators (default: as many as the registers allow)")
    p.add_argument("--load-regs", type=int, default=4,
                   help="registers the loads go to in turn - no FMA reads them")
    p.add_argument("--iterations", type=int, default=2000,
                   help="loop iterations a timing runs at least")
    p.add_argument("--min-fmas", type=int, default=1000000,
                   help="FMAs a timing runs at least: short bodies get more iterations")
    p.add_argument("--measurements", type=int, default=10,
                   help="timings each: the least counts, the median shows whether they agree")
    p.add_argument("--spread", type=float, default=2.0,
                   help="percent the median may exceed the least before a result is marked "
                        "unsteady (~)")
    p.add_argument("--loop-offset", type=int, default=None,
                   help="the loops' heads this many bytes past a 64-byte boundary")
    p.add_argument("--show-asm", action="store_true", help="print each loop")
    add_common_args(p)
    args = p.parse_args()

    ratios = [int(r) for r in args.ratios.split(",") if r.strip()]
    dts = [d.strip() for d in args.dt.split(",") if d.strip()]
    forms = [f.strip() for f in args.form.split(",") if f.strip()]
    if any(r < 0 for r in ratios) or any(d not in EBYTES for d in dts) \
            or any(f not in ("vf", "vv") for f in forms):
        sys.exit("--ratios are 0 or more, --dt FP64/FP32/FP16, --form vf/vv")
    if "FP16" in dts and args.isa.startswith("rv"):
        flags = fp16_march(args.cflags)
        if flags != args.cflags:
            print(f"# FP16: --cflags {flags!r}", file=sys.stderr)
            args.cflags = flags

    exps, texts = [], []          # (name, dt, form, ratio, accumulators, fmas, loads)
    simd_c = None
    for dt in dts:
        params = isa_params(args.isa_param)
        if "FP16" == dt and args.isa.startswith("rv"):
            params.setdefault("ZVFH", "on")
        ops = isa_ops(args.isa, dt, params)
        simd = ops.gen.c_simd_size_function
        simd_c = simd_c or (simd() if callable(simd) else simd)
        free = free_vregs(ops)
        for form in forms:
            if "vf" == form and ops.fma_form is None:
                print(f"# {args.isa} has no FMA with a scalar operand: vf runs as vv",
                      file=sys.stderr)
            # the sources: a vector, and a scalar operand (a vector register
            # where scalars live in them) or a second vector
            sources = 1 + (1 if "vv" == form or "vreg" == ops.scalar_tag else 0)
            room = free - sources - (args.load_regs if any(ratios) else 0)
            n = args.accumulators or room
            if not 1 <= n <= room:
                sys.exit(f"{args.isa} {dt} {form}: {free} vector registers - {sources} sources"
                         f"{f', {args.load_regs} load targets' if any(ratios) else ''}: room for "
                         f"1 to {room} accumulators, not {n}")
            for r in ratios:
                acc = n
                while r and math.lcm(acc, r) > MAX_BODY:
                    acc -= 1          # a body of lcm(N, R) FMAs, kept to MAX_BODY
                if acc != n:
                    print(f"# {dt} {form} ratio {r}: {acc} accumulators (lcm with {n} would make "
                          f"a body over {MAX_BODY} FMAs)", file=sys.stderr)
                body, fmas, loads = body_for(form, acc, r, args.load_regs)
                name = f"{dt}.{form}.r{r}"
                text, loop = ops.function(f"fl_{len(exps)}", body, name, None, args.loop_offset)
                texts.append(text)
                exps.append((name, dt, form, r, acc, fmas, loads))
                if args.show_asm:
                    print(f"# {name}: {acc} accumulators, {fmas} FMAs, {loads} loads\n{loop}")

    out = args.out or f"ukrfmaload-{args.isa}"
    os.makedirs(out, exist_ok=True)
    asm = os.path.join(out, "fmaload.S")
    with open(asm, "w") as f:
        f.write("\n".join(texts))
    simd_c = simd_c.replace("inline size_t get_simd_size", "static inline size_t get_simd_size")
    externs = "".join(f"extern void fl_{i}(long, double*);\n" for i in range(len(exps)))
    calls = "".join(f'    fill({EBYTES[dt]});\n    measure("{name}", fl_{i}, iters(n, {fmas}), reps);\n'
                    for i, (name, dt, form, r, acc, fmas, loads) in enumerate(exps))
    driver = (C_HEADER + load_prologue(args.use_prologue) + simd_c + "\n" + externs
              + f"#define BUF_DOUBLES {BUF_BYTES // 8}\n" + MEASURE + FILL + C_MAIN_BEGIN
              + '    printf("# simd_bytes %zu\\n", (size_t)get_simd_size());\n'
              + "    fflush(stdout);\n"
              + "    long n = argc > 1 ? atol(argv[1]) : 2000;\n"
              + "    int reps = argc > 2 ? atoi(argv[2]) : 10;\n"
              + "    min_fmas = argc > 3 ? atol(argv[3]) : 1000000;\n"
              + calls + C_MAIN_END)
    exe = build(args, out, driver, [asm], "ukrfmaload_bench")
    argv = [str(args.iterations), str(args.measurements), str(args.min_fmas)]
    if args.build_only:
        print(f"built {exe}: run '{exe} {' '.join(argv)}' on the target")
        return

    outputs = run(args, exe, argv)
    values, unit = parse(outputs)
    simd_bytes = None
    for text in outputs.values():
        m = re.search(r"^# simd_bytes (\d+)", text, re.M)
        if m:
            simd_bytes = int(m.group(1))
            break
    if not simd_bytes:
        sys.exit("the benchmark didn't report its vector length")
    medians = {}
    for cpu, text in outputs.items():
        seen = {}
        for line in text.splitlines():
            w = line.split()
            if 3 == len(w) and "median" == w[0]:
                seen.setdefault(w[1], []).append(float(w[2]))
        medians[cpu] = {k: sorted(v)[len(v) // 2] for k, v in seen.items()}
    cpus = list(values)
    per = "cycle" if "cycles" == unit else unit
    unsteady = 0
    print(f"FLOP per {per}; vectors of {simd_bytes} bytes at run time; FMAs per load exact, "
          f"across the loop's edge too")
    head = [f"core {c}" if c is not None else per for c in cpus]
    print(" | ".join(["dt   form ratio", "acc", "body", *[f"{h:>8}" for h in head], "of baseline"]))
    base = {}
    for name, dt, form, r, acc, fmas, loads in exps:
        flop = fmas * 2 * simd_bytes // EBYTES[dt]
        rates, marks = [], []
        for c in cpus:
            cyc = values[c].get(name, float("nan"))
            rates.append(flop / cyc if cyc and cyc == cyc else float("nan"))
            med = medians.get(c, {}).get(name)
            shaky = bool(cyc and cyc == cyc and med and med > cyc * (1 + args.spread / 100))
            marks.append("~" if shaky else " ")
            unsteady += shaky
        if 0 == r:
            base[(dt, form)] = rates
        b = base.get((dt, form))
        share = (", ".join(f"{100 * x / y:5.1f}%" for x, y in zip(rates, b) if y == y and x == x)
                 if b and r else "")
        cells = [f"{x:8.3f}{m}" if x == x else f"{'crashed':>8} " for x, m in zip(rates, marks)]
        print(" | ".join([f"{dt:4} {form:4} {r:5d}", f"{acc:3d}", f"{fmas:4d}", *cells, share]))

    if unsteady:
        print(f"~ unsteady: the timings' median over {args.spread:g}% above their least - the "
              f"core ran this loop at more than one speed; the least is its fastest")


if __name__ == "__main__":
    main()

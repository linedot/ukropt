"""
ukropt-sweep's any-kernel mode: a built kernel - gemm, packm - called as it
is, through its own symbol, its parameters given from what it says of
itself: the loop's count (k, n - in vector lengths where its loop counts
them), each component's base and strides (bytes) from the family's layouts
('m:c', 'n:s', 'k:p': the first index contiguous, each after it strided by
the extents before - packed tightly) and its sizes (m: MR or cdim, n: NR,
the loop's index: its count). Timed at loop counts - a line fitted, as for
gemm's k - or at working sets: the loop count whose components fill a
level's bytes (L1=32K,L2=1M,L3=16M,MEM=1G), its rate in bytes a cycle.
"""
import copy
import statistics

from ukrgen.injectors.blis.families import configuration, build_gemm, build_packm, key_params
from ukrgen.injectors.blis.cgen import asm_source, asm_symbol, vl_helpers, c_expr
from ukrgen.flow.isa_map import make_generator

CTYPE = {"FP64": "double", "FP32": "float"}

ANY_DRIVER = r"""
#ifndef UKRSWEEP_PROLOGUE
#define UKRSWEEP_PROLOGUE(argc, argv) ((void)0)
#endif
typedef void (*callf)(dim_t, WT**);
typedef size_t (*footf)(dim_t);
typedef size_t (*countf)(int, dim_t);
static double bytes_of(const char* s)
{
    char* end; double v = strtod(s, &end);
    if (*end == 'K' || *end == 'k') v *= 1024.0;
    else if (*end == 'M' || *end == 'm') v *= 1024.0 * 1024.0;
    else if (*end == 'G' || *end == 'g') v *= 1024.0 * 1024.0 * 1024.0;
    return v;
}
static double best_any(callf f, dim_t L, WT** bufs, int reps, int iters)
{
    double t = 1e300;
    for (int r = 0; r < reps; ++r) {
#ifdef USE_PCLIB
        pc_tic();
        for (int i = 0; i < iters; ++i) f(L, bufs);
        double d = (double)pc_toc() / iters;
#else
        double t0 = now();
        for (int i = 0; i < iters; ++i) f(L, bufs);
        double d = (now() - t0) / iters;
#endif
        if (d < t) t = d;
    }
    return t;
}
/* a variant at each request: a loop count, or NAME=BYTES - the count whose
   working set fills them; buffers for the largest */
static void measure(int v, int o, callf f, footf foot, countf count, int nbufs, int nreq, char** req, int reps)
{
    dim_t Ls[64]; int named[64]; int n = 0; dim_t Lmax = 1;
    size_t f0 = foot(0), step = foot(1) - foot(0);
    for (int r = 0; r < nreq && n < 64; ++r) {
        const char* eq = strchr(req[r], '=');
        named[n] = eq != NULL;
        if (eq) { double b = bytes_of(eq + 1); Ls[n] = (step && b > (double)f0) ? (dim_t)((b - (double)f0) / (double)step) : 1; }
        else Ls[n] = atol(req[r]);
        if (Ls[n] < 1) Ls[n] = 1;
        if (Ls[n] > Lmax) Lmax = Ls[n];
        ++n;
    }
    WT* bufs[16];
    for (int j = 0; j < nbufs && j < 16; ++j) {
        size_t c = count(j, Lmax) + 64, bytes = (c * sizeof(WT) + 255) / 256 * 256;
        bufs[j] = aligned_alloc(256, bytes);
        for (size_t e = 0; e < c; ++e) bufs[j][e] = (WT)((e % 7) * 0.01);
    }
    for (int r = 0; r < n; ++r) {
        size_t bytes = foot(Ls[r]);
        /* a level's call may move a gigabyte: fewer of them; loop counts as
           given, every one with ITERS calls (a line fitted through them) */
        double scaled = (double)ITERS * 32768.0 / (double)(bytes ? bytes : 1);
        int iters = !named[r] || scaled >= ITERS ? ITERS : (scaled < 1 ? 1 : (int)scaled);
        double t = best_any(f, Ls[r], bufs, reps, iters);
        if (named[r])
            printf("level %d %d %.*s %ld %.2f %zu\n", v, o, (int)(strchr(req[r], '=') - req[r]), req[r],
                   (long)Ls[r], t, bytes);
        else
            printf("variant %d %d %ld %.2f\n", v, o, (long)Ls[r], t);
    }
    fflush(stdout);
}
int main(int argc, char** argv)
{
    UKRSWEEP_PROLOGUE(argc, argv);
    const char *cpu = getenv("UKRSWEEP_CPU");
    if (cpu && *cpu) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(atoi(cpu), &set);
        if (sched_setaffinity(0, sizeof set, &set))
            perror("sched_setaffinity");
    }
#ifdef USE_PCLIB
#define OPEN_COUNTER() pc_init(PC_EVENT)
    printf("# unit %s\n", PC_EVENT);
#else
#define OPEN_COUNTER() counter_open()
    counter_open();
    if (cycles_fd >= 0) { printf("# unit cycles\n"); close(cycles_fd); cycles_fd = -1; }
    else printf("# unit ns\n");
#endif
    fflush(stdout);
VARIANTS
    return 0;
}
"""


def layouts_of(params: dict) -> dict:
    """every component's layout: X-layout ['n:c', 'm:s'] - [('n', 'c'), ('m', 's')]"""
    return {key[:-len("-layout")]: [tuple(entry.split(":")) for entry in value]
            for key, value in params.items() if key.endswith("-layout") and value}


def kernel_c(info: dict, sym: str, dt: str) -> dict:
    """
    A kernel's C: its declaration, a call taking the loop's count and the
    components' buffers, its working set's bytes and each buffer's elements
    for a count - from its parameters, layouts and sizes (info)
    """
    ctype = CTYPE[dt]
    layouts = info["layouts"]
    loops = [p for p in info["params"] if "(" not in p]
    if 1 != len(loops):
        raise ValueError(f"{sym}: not one loop count among {info['params']}")
    loop = loops[0]
    unit = f"{info['vl_fn']}( sizeof( WT ) )" if "VL" == info["loop_unit"] else "1"
    extent = {loop: f"((dim_t)L * {unit})"}
    for index, size in (("m", info["mr"]), ("n", info["nr"])):
        if index != loop:
            extent[index] = f"((dim_t)({c_expr(size, ctype, info['vl_fn'])}))"
    comps = [c for c in layouts if f"base({c})" in info["params"]]

    def along(index):
        if index not in extent:
            raise ValueError(f"{sym}: no extent for index {index} (m, n and the loop's are known)")
        return extent[index]

    def count(c):
        return " * ".join(along(i) for i, _ in layouts[c])

    def stride(c, index):
        before = []
        for i, _ in layouts[c]:
            if i == index:
                return " * ".join(before) or "1"
            before.append(along(i))
        raise ValueError(f"{sym}: {c} has no index {index}")

    types, args = [], []
    for p in info["params"]:
        if p == loop:
            types.append("dim_t")
            args.append("L")
        elif p.startswith("base("):
            c = p[5:-1]
            types.append("void*")
            args.append(f"bufs[{comps.index(c)}]" if c in comps else "(void*)&one")
        elif p.startswith("stride("):
            c, index = p[7:-1].split(",")
            types.append("inc_t")
            args.append(f"(inc_t)({stride(c, index)}) * (inc_t)sizeof(WT)")
        else:
            raise ValueError(f"{sym}: no argument for its parameter {p}")
    counts = [f"(size_t)({count(c)})" for c in comps]
    cases = " ".join(f"case {j}: return {n};" for j, n in enumerate(counts))
    return {"decl": f"void {sym}( {', '.join(types)} );\n",
            "funcs": f"static void call_{sym}( dim_t L, WT** bufs ) {{ {sym}( {', '.join(args)} ); }}\n"
                     f"static size_t foot_{sym}( dim_t L ) {{ return ({' + '.join(counts) or '0'}) * sizeof(WT); }}\n"
                     f"static size_t count_{sym}( int j, dim_t L ) {{ switch (j) {{ {cases} }} return 0; }}\n",
            "nbufs": len(comps), "loop": loop}


def apply_settings(kind: str, params: dict, settings: dict):
    """a family key maps to its flow parameter, anything else is one; null removes it"""
    for key, value in settings.items():
        mapped = key_params(kind, {key: value if value is not None else "x"})
        names = list(mapped) or [key]
        if value is None:
            for name in names:
                params.pop(name, None)
        elif mapped:
            params.update(mapped)
        else:
            params[key] = value


def build_any_variant(job):
    """
    One variant of any kind, in a worker process: its assembly, vector-length
    helpers and what kernel_c needs (picklable), or why it didn't build
    """
    i, settings, config, kind, dtype, family_name, case, size = job
    cfg = configuration.from_files({(kind, dtype): config})
    base = next(f for f in cfg.families if f.name == family_name)
    fam = copy.deepcopy(base)
    fam.name = f"{base.name}{i}"
    try:
        if "gemm" == kind:
            m, n = size
            fam.sizes = {"m": [m], "n": [n]}
            apply_settings(kind, fam.params, settings)
            ok, skipped = build_gemm(cfg.isa, cfg.name, fam)
            params = fam.params
        else:
            m = size[0]
            chosen = [c for c in fam.cases if c[0] == case and m in c[1]]
            if not chosen:
                return {"i": i, "settings": settings, "error": f"no {case} case of {m} rows in {family_name}"}
            kind_, _, cparams = chosen[0]
            cparams = dict(cparams)
            apply_settings(kind, cparams, settings)
            fam.cases = [(kind_, [m], cparams)]
            ok, skipped = build_packm(cfg.isa, cfg.name, fam)
            params = dict(fam.params, **cparams)
    except Exception as err:            # a variant's failure is its own
        return {"i": i, "settings": settings, "error": str(err)[:200]}
    if not ok:
        return {"i": i, "settings": settings, "error": str(skipped[0][2])[:200]}
    k = ok[0]
    return {"i": i, "settings": settings, "asm": asm_source(k), "symbol": k.symbol, "asm_symbol": asm_symbol(k),
            "helpers": vl_helpers([k], make_generator(cfg.isa)),
            "info": {"params": list(k.params), "mr": k.mr, "nr": k.nr, "vl_fn": k.vl_fn,
                     "loop_unit": k.loop_unit, "layouts": layouts_of(params)}}


def any_driver(built: list, dtype: str, offsets: list, reps: int) -> tuple[str, str, list]:
    """
    The benchmark's declarations and functions (before the driver), its
    VARIANTS (a child each) and the assembly sources' renamings: (head,
    variants, [(variant, offset, symbol)])
    """
    head, calls, placed = [], [], []
    for r in built:
        for o in offsets:
            # the copy's symbols: the kernel's renamed, and the one its
            # assembly defines (its own, or _asm after it)
            sym = f"{r['symbol']}_a{o}"
            c = kernel_c(r["info"], r["asm_symbol"].replace(r["symbol"], sym), dtype)
            head.append(c["decl"] + c["funcs"])
            placed.append((r, o, sym))
            fn = r["asm_symbol"].replace(r["symbol"], sym)
            calls.append(f'    {{ pid_t p = fork(); if (0 == p) {{ OPEN_COUNTER(); measure({r["i"]}, {o}, '
                         f'call_{fn}, foot_{fn}, count_{fn}, {c["nbufs"]}, argc - 1, argv + 1, {reps}); '
                         f'_exit(0); }} int st = 0; waitpid(p, &st, 0); if (!WIFEXITED(st) || WEXITSTATUS(st)) '
                         f'{{ printf("crashed {r["i"]} {o} %d\\n", WIFSIGNALED(st) ? WTERMSIG(st) : '
                         f'WEXITSTATUS(st)); fflush(stdout); }} }}')
    return "".join(head), "\n".join(calls), placed


def report_levels(runs: dict, settings: dict, unit: str):
    """
    Per variant and level: the loop count, the time a loop step and the
    bytes a unit of time - each the fastest placement, the median of runs -
    ranked by the first level's time a step
    """
    import re
    per = {}
    for cpu, text in runs.items():
        for i, o, name, L, t, nbytes in re.findall(r"level (\d+) (\d+) (\S+) (\d+) ([-\d.]+) (\d+)", text):
            per.setdefault((int(i), name), {}).setdefault(int(o), []).append((int(L), float(t), int(nbytes)))
    if not per:
        raise SystemExit("the benchmark printed no level")
    names = list(dict.fromkeys(name for _, name in per))
    table = {}
    for (i, name), by_offset in per.items():
        best = min((statistics.median(t for _, t, _ in pts), pts[0][0], pts[0][2])
                   for pts in by_offset.values())
        table.setdefault(i, {})[name] = best
    first = names[0]
    rows = sorted(table, key=lambda i: table[i][first][0] / max(1, table[i][first][1])
                  if first in table[i] else float("inf"))
    per = "cycle" if "cycles" == unit else unit
    print(f"{unit} a loop step and bytes a {per}, per level "
          f"({', '.join(names)}) | settings")
    for i in rows:
        cols = []
        for name in names:
            if name not in table[i]:
                cols.append(f"{name}: -")
                continue
            t, L, nbytes = table[i][name]
            cols.append(f"{name}: {t / L:8.2f} ({L}) {nbytes / t:7.2f} B")
        print(f"{' | '.join(cols)} | {settings[i]}")
    return rows

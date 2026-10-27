"""
ukropt-sweep's BLIS-like packm mode: a packm family called as BLIS calls it
- through ukrgen's dispatch, with BLIS's arguments as given (cdim, inca,
lda, ldp, ...) - which picks the case those fall into, or the reference.
The sweep says which (a case, the reference, or a case and the reference
for what it leaves), checks the panel against the reference's and times
the call at numbers of columns n.
"""
import copy
import re
import statistics

from ukrgen.injectors.blis.families import configuration, build_packm
from ukrgen.injectors.blis.cgen import packm_dispatch, asm_source, asm_symbol, vl_helpers
from ukrgen.flow.isa_map import make_generator

from .anykernel import apply_settings

#: BLIS's packm arguments the mode takes, strides in elements as BLIS's (1:
#: the next element the next one); n_max: n plus extra
KEYS = ("cdim", "cdim_max", "inca", "lda", "ldp", "n_extra", "kappa")

BLIS_DRIVER = r"""
#ifndef UKRSWEEP_PROLOGUE
#define UKRSWEEP_PROLOGUE(argc, argv) ((void)0)
#endif
typedef void (*packf)( conj_t, pack_t, dim_t, dim_t, dim_t, dim_t, dim_t, const void*, const void*,
                       inc_t, inc_t, void*, inc_t, const void*, const cntx_t* );
static double best_blis(packf f, dim_t n, dim_t inca, dim_t lda, const WT* A, WT* P, int reps, int iters)
{
    const WT kappa = KAPPA; double t = 1e300;
    for (int r = 0; r < reps; ++r) {
#ifdef USE_PCLIB
        pc_tic();
        for (int i = 0; i < iters; ++i)
            f(0, 0, CDIM, CDIM_MAX, 1, n, n + N_EXTRA, &kappa, A, inca, lda, P, LDP, NULL, NULL);
        double d = (double)pc_toc() / iters;
#else
        double t0 = now();
        for (int i = 0; i < iters; ++i)
            f(0, 0, CDIM, CDIM_MAX, 1, n, n + N_EXTRA, &kappa, A, inca, lda, P, LDP, NULL, NULL);
        double d = (now() - t0) / iters;
#endif
        if (d < t) t = d;
    }
    return t;
}
/* a variant at each n: which case it falls into (and how often the
   reference is called), whether its panel is the reference's, its time */
static void measure_blis(int v, packf f, int nreq, char** req, int reps)
{
    dim_t ns[64]; int n = 0; dim_t nmax = 1;
    for (int r = 0; r < nreq && n < 64; ++r) { ns[n] = atol(req[r]); if (ns[n] < 1) ns[n] = 1; if (ns[n] > nmax) nmax = ns[n]; ++n; }
    size_t na = (size_t)(CDIM - 1) * INCA + (size_t)(nmax - 1) * LDA + 1 + 64;
    size_t np = (size_t)LDP * (nmax + N_EXTRA) + 64;
    WT* A = aligned_alloc(256, (na * sizeof(WT) + 255) / 256 * 256);
    WT* P = aligned_alloc(256, (np * sizeof(WT) + 255) / 256 * 256);
    WT* R = aligned_alloc(256, (np * sizeof(WT) + 255) / 256 * 256);
    for (size_t e = 0; e < na; ++e) A[e] = (WT)((e % 13) - 6);
    for (int r = 0; r < n; ++r) {
        const dim_t nn = ns[r], inca = INCA, lda = LDA;
        const WT kappa = KAPPA;
        for (size_t e = 0; e < np; ++e) P[e] = R[e] = (WT)-99;
        ukrsweep_ref(0, 0, CDIM, CDIM_MAX, 1, nn, nn + N_EXTRA, &kappa, A, inca, lda, R, LDP, NULL, NULL);
        ukrsweep_case = -1; ukrsweep_refs = 0;
        f(0, 0, CDIM, CDIM_MAX, 1, nn, nn + N_EXTRA, &kappa, A, inca, lda, P, LDP, NULL, NULL);
        int chosen = ukrsweep_case, refs = ukrsweep_refs; long wrong = 0;
        for (dim_t j = 0; j < nn + N_EXTRA; ++j)
            for (dim_t i = 0; i < CDIM_MAX; ++i)
                if (P[i + j * LDP] != R[i + j * LDP]) ++wrong;
        double t = best_blis(f, nn, inca, lda, A, P, reps, ITERS);
        printf("blis %d %ld %.2f %d %d %ld %ld %ld\n", v, (long)nn, t, chosen, refs, wrong, (long)inca, (long)lda);
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

#: the reference every variant's dispatch calls (by its own name, defined to
#: this one), counting its calls; the case the dispatch called last
REFERENCE = r"""
typedef int conj_t;
typedef int pack_t;
static int ukrsweep_case = -1, ukrsweep_refs = 0;
void ukrsweep_ref( conj_t conja, pack_t schema, dim_t cdim, dim_t cdim_max, dim_t cdim_bcast, dim_t n,
                   dim_t n_max, const void* kappa, const void* a, inc_t inca, inc_t lda, void* p, inc_t ldp,
                   const void* params, const cntx_t* cntx )
{
    const WT k = *( const WT* )kappa; const WT* A = a; WT* P = p;
    ukrsweep_refs++;
    for ( dim_t j = 0; j < n_max; j++ ) for ( dim_t i = 0; i < cdim_max; i++ )
        P[ i + j * ldp ] = ( i < cdim && j < n ) ? k * A[ i * inca + j * lda ] : 0;
}
"""


def parse_call(text: str) -> dict:
    """
    --blis cdim=8,inca=1,lda=1024,ldp=8: BLIS's arguments, strides in
    elements as BLIS's. cdim needed; by default cdim_max cdim, ldp cdim_max,
    inca 1, lda cdim where inca is 1 (the columns adjacent, as tight as they
    go), n_extra 0 (n_max = n), kappa 1
    """
    args = {"cdim": None, "cdim_max": None, "inca": 1, "lda": None, "ldp": None, "n_extra": 0, "kappa": 1.0}
    for part in (p for p in text.split(",") if p.strip()):
        key, _, value = part.partition("=")
        key = key.strip()
        if key not in KEYS:
            raise SystemExit(f"--blis: {key}? ({', '.join(KEYS)})")
        args[key] = float(value) if "kappa" == key else int(value)
    if not args["cdim"]:
        raise SystemExit("--blis needs cdim")
    args["cdim_max"] = args["cdim_max"] or args["cdim"]
    args["ldp"] = args["ldp"] or args["cdim_max"]
    if args["lda"] is None:
        if 1 != args["inca"]:
            raise SystemExit("--blis: lda - where inca isn't 1, the columns' stride isn't cdim")
        args["lda"] = args["cdim"]
    if args["cdim_max"] < args["cdim"] or args["ldp"] < args["cdim_max"]:
        raise SystemExit("--blis: cdim <= cdim_max <= ldp")
    if args["inca"] < 1 or args["lda"] < 1:
        raise SystemExit("--blis: inca and lda are strides of at least 1 element (1: the next element)")
    return args


def check_call(call: dict, ns: list):
    """
    The elements apart for every n timed: the columns adjacent (inca 1) at
    least cdim apart, the rows adjacent (lda 1) at least the most columns
    apart; else one stride nesting the other
    """
    n = max(ns)
    inca, lda, cdim = call["inca"], call["lda"], call["cdim"]
    if 1 == inca and lda < cdim:
        raise SystemExit(f"--blis: lda {lda} - the columns overlap; at least cdim, {cdim}")
    if 1 == lda and 1 != inca and inca < n:
        raise SystemExit(f"--blis: inca {inca} - the rows overlap at {n} columns; at least {n}")
    if 1 != inca and 1 != lda and lda < cdim * inca and inca < n * lda:
        raise SystemExit(f"--blis: inca {inca}, lda {lda} - the elements overlap at {n} columns")


def instrument(dispatch: str, built: list, name: str, ch: str) -> str:
    """
    The dispatch, each kernel's call recording its case (outside the
    declarations), the reference by this name the one counting its calls
    """
    out = []
    for line in dispatch.split("\n"):
        if not line.lstrip().startswith("void "):
            for j, k in enumerate(built):
                sym = asm_symbol(k)
                line = line.replace(f"{sym}( ", f"( ukrsweep_case = {j}, {sym} )( ")
        out.append(line)
    return f"#define bli_{ch}{ch}packm_{name}_ref ukrsweep_ref\n" + "\n".join(out) + \
        f"\n#undef bli_{ch}{ch}packm_{name}_ref\n"


def build_blis_variant(job):
    """
    A packm family's every case and size, as blisgen builds them - its
    variant's settings over every case - with its dispatch, in a worker
    process; or why it didn't build
    """
    i, settings, config, dtype, family_name, keep = job
    cfg = configuration.from_files({("packm", dtype): config})
    base = next(f for f in cfg.families if f.name == family_name)
    fam = copy.deepcopy(base)
    name = f"{cfg.name}v{i}"
    cases = []
    for kind, ms, params in fam.cases:
        if keep and kind not in keep:
            continue
        params = dict(params)
        apply_settings("packm", params, settings)
        cases.append((kind, ms, params))
    fam.cases = cases
    try:
        built, skipped = build_packm(cfg.isa, name, fam)
    except Exception as err:            # a variant's failure is its own
        return {"i": i, "settings": settings, "error": str(err)[:200]}
    if not built:
        return {"i": i, "settings": settings, "error": str(skipped[0][2])[:200] if skipped else "nothing built"}
    ch = {"FP64": "d", "FP32": "s"}[dtype]
    return {"i": i, "settings": settings, "name": name, "ch": ch,
            "asm": [(asm_symbol(k), asm_source(k)) for k in built],
            "dispatch": instrument(packm_dispatch(name, dtype, built), built, name, ch),
            "helpers": vl_helpers(built, make_generator(cfg.isa)),
            # each case as its symbol ends: its kind and rows (vec8, trans4)
            "cases": [k.symbol.rsplit("_", 1)[1] for k in built],
            "skipped": [f"{s[1]}: {str(s[2])[:80]}" for s in skipped]}


def blis_driver(built: list, call: dict, reps: int) -> tuple[str, str]:
    """the benchmark's definitions before the driver and its VARIANTS"""
    defines = "".join(f"#define {k.upper()} {call[k]}\n" for k in ("cdim", "cdim_max", "inca", "lda", "ldp"))
    defines += f"#define N_EXTRA {call['n_extra']}\n#define KAPPA (WT){call['kappa']}\n"
    calls = [f'    {{ pid_t p = fork(); if (0 == p) {{ OPEN_COUNTER(); measure_blis({r["i"]}, '
             f'bli_{r["ch"]}packm_{r["name"]}, argc - 1, argv + 1, {reps}); _exit(0); }} int st = 0; '
             f'waitpid(p, &st, 0); if (!WIFEXITED(st) || WEXITSTATUS(st)) {{ printf("crashed {r["i"]} 0 %d\\n", '
             f'WIFSIGNALED(st) ? WTERMSIG(st) : WEXITSTATUS(st)); fflush(stdout); }} }}' for r in built]
    return defines + REFERENCE + "".join(r["dispatch"] for r in built), "\n".join(calls)


def report_blis(runs: dict, built: list, call: dict, unit: str, esize: int):
    """
    Per variant: the case each n fell into, the reference's calls, wrong
    elements, the time a call (the fastest, the median of runs) and a column,
    and bytes (A's read, P's written) a unit of time
    """
    by = {r["i"]: r for r in built}
    per = {}
    for _, text in runs.items():
        for v, n, t, case, refs, wrong, inca, lda in re.findall(
                r"blis (\d+) (\d+) ([-\d.]+) (-?\d+) (\d+) (\d+) (\d+) (\d+)", text):
            per.setdefault(int(v), {}).setdefault(int(n), []).append(
                (float(t), int(case), int(refs), int(wrong), int(inca), int(lda)))
    if not per:
        raise SystemExit("the benchmark printed no packm call")
    unit_of = "cycle" if "cycles" == unit else unit
    print(f"BLIS packm: cdim {call['cdim']} of {call['cdim_max']}, ldp {call['ldp']}, kappa {call['kappa']}, "
          f"n_max n + {call['n_extra']}")
    bad = False
    for v in sorted(per):
        r = by[v]
        print(f"variant {v} {r['settings']}:")
        for n, pts in sorted(per[v].items()):
            t = statistics.median(p[0] for p in pts)
            case, refs, wrong, inca, lda = pts[0][1:]
            which = r["cases"][case] if case >= 0 else "-"
            how = (f"{which}" + (f" + reference x{refs}" if refs else "")) if case >= 0 else \
                f"reference x{refs}"
            nbytes = (call["cdim"] * n + call["cdim_max"] * (n + call["n_extra"])) * esize
            bad = bad or wrong > 0
            print(f"  n {n:6} (inca {inca}, lda {lda}): {how:28} {t:10.1f} {unit} a call, "
                  f"{t / n:7.2f} a column, {nbytes / t:7.2f} B a {unit_of}"
                  + (f" - {wrong} WRONG" if wrong else ""))
    if bad:
        print("WRONG: a panel differs from the reference's")
    return not bad

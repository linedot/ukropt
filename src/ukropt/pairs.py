"""
ukropt-pairs: how a core executes short instruction mixes. Microbenchmarks
generated through asmgen, so an experiment written once runs on every ISA
asmgen has.

An experiment is a loop body of ISA-neutral roles, repeated:

    fma  C A S       C += A * S, S a scalar operand: RVV vfmacc.vf, NEON
                     fmla by element, SVE fmla by block lane. x86 has no
                     scalar form: S is a vector there (as after a broadcast)
    vfma C A B       C += A * B, vector by vector
    vload V [P]      a vector load: from pointer P, else the buffer's start
    sload S OFF [P]  a scalar operand loaded (fld, ldr d, vmovsd) at OFF
                     bytes from pointer P, else from the buffer's start
    iadd G IMM       G += IMM (a pointer moves this way)

Registers are named by class: c (accumulators), a b (vector sources) and
v (vector load targets) are vector registers; s t (scalar operands) are FP
registers on RVV and vector registers elsewhere; g are general registers;
p are pointers into the buffer (2 MB), set before the loop to the buffer
plus the experiment's `pointers: {p0: OFFSET}` (0 if not given). Each name
is a register of its own, allocated through asmgen - an experiment needing
more than an ISA has is reported, not miscompiled.

{...} in an instruction is a Python expression of i (the repetition), of
C, S, T - the accumulators and scalar operands the ISA has room for (16,
8, 4 with 32 vector registers; 8, 4, 2 with 16) - and of VB, a vector's
bytes (fixed on NEON and x86, --vector-bytes on RVV and SVE). A body's
entry is a template, or {op: TEMPLATE, every: N} - emitted where
i % N == N - 1 - or {op: TEMPLATE, when: EXPRESSION of i}. Pointer offsets
may be templates as well.

Each experiment is a function pairs_N(long n, double *buf) running its loop
n times; the driver times n and 2n iterations and keeps the difference -
the call's cost gone - per iteration, the least of --measurements timings.
"""

import argparse
import fnmatch
import os
import re
import sys

import yaml

from .harness import (C_HEADER, C_MAIN_BEGIN, C_MAIN_END, add_common_args, build,
                      load_prologue, run)

# name: what it shows, body, repetitions
BUILTIN = {
    "fma": ("independent FMAs, all from one vector source",
            ["fma c{i%C} a0 s{i%S}"], 32),
    "fma-2src": ("independent FMAs, vector sources alternating",
                 ["fma c{i%C} a{i%2} s{i%S}"], 32),
    "fma-4src": ("independent FMAs, four vector sources in turn",
                 ["fma c{i%C} a{i%4} s{i%S}"], 32),
    "fma-1s": ("independent FMAs, sources alternating, one scalar operand",
               ["fma c{i%C} a{i%2} s0"], 32),
    "vfma-2src": ("independent vector x vector FMAs, sources alternating",
                  ["vfma c{i%C} a{i%2} b{i%2}"], 32),
    "chain-1": ("one dependency chain: cycles per FMA = the latency",
                ["fma c0 a0 s0"], 32),
    "chain-2": ("two chains", ["fma c{i%2} a0 s0"], 32),
    "chain-4": ("four chains", ["fma c{i%4} a0 s0"], 32),
    "chain-8": ("eight chains", ["fma c{i%8} a0 s0"], 32),
    "fma+sload": ("an FMA and a scalar load in turn (loaded registers unread)",
                  ["fma c{i%C} a{i%2} s{i%S}", "sload t{i%T} {8*(i%T)}"], 32),
    "fma2+sload": ("two FMAs per scalar load",
                   ["fma c{i%C} a{i%2} s{i%S}", {"op": "sload t{i%T} {8*(i%T)}", "every": 2}], 32),
    "fma+sload-war": ("an FMA and a scalar load in turn, loads into the operands, read S/2 FMAs on",
                      ["fma c{i%C} a{i%2} s{i%S}", "sload s{(i+S//2)%S} {8*(i%S)}"], 32),
    # the loads' targets alternate: no load waits for the previous one's
    # write to the same register
    "fma+vload/2": ("a vector load per two FMAs (loaded registers unread)",
                    ["fma c{i%C} a{i%2} s{i%S}", {"op": "vload v{(i//2)%2}", "every": 2}], 32),
    "fma+vload/4": ("a vector load per four FMAs",
                    ["fma c{i%C} a{i%2} s{i%S}", {"op": "vload v{(i//4)%2}", "every": 4}], 32),
    "fma+vload/8": ("a vector load per eight FMAs",
                    ["fma c{i%C} a{i%2} s{i%S}", {"op": "vload v{(i//8)%2}", "every": 8}], 32),
    "fma+vload/16": ("a vector load per sixteen FMAs",
                     ["fma c{i%C} a{i%2} s{i%S}", {"op": "vload v{(i//16)%2}", "every": 16}], 32),
    "fma+vload/2-1reg": ("as fma+vload/2, every load into one register: loads chained on it",
                         ["fma c{i%C} a{i%2} s{i%S}", {"op": "vload v0", "every": 2}], 32),
    # groups of four FMAs reading one vector source, a load after each group:
    # where it lands against its readers
    "vload-raw-near": ("a group of 4 FMAs reads the vector loaded right before it (RAW: 1 instr.)",
                       ["fma c{i%C} v{(i//4)%2} s{i%S}", {"op": "vload v{(i//4+1)%2}", "every": 4}], 32),
    "vload-war-near": ("a load overwrites the vector its group of 4 FMAs just read (WAR: 1 instr.)",
                       ["fma c{i%C} v{(i//4)%2} s{i%S}", {"op": "vload v{(i//4)%2}", "every": 4}], 32),
    "vload-far": ("as the two above, load and readers two groups apart both ways",
                  ["fma c{i%C} v{(i//4)%4} s{i%S}", {"op": "vload v{(i//4+2)%4}", "every": 4}], 32),
    "fma+iadd": ("an FMA and an integer add in turn",
                 ["fma c{i%C} a{i%2} s{i%S}", "iadd g{i%2} 8"], 32),
}

KINDS = {"fma": (3,), "vfma": (3,), "vload": (1, 2), "sload": (2, 3), "iadd": (2,)}
BUF_BYTES = 2 << 20
FIXED_VB = {"neon": 16, "avx256": 32, "avx2": 32, "avx512": 64}
TEMPLATE = re.compile(r"\{([^{}]+)\}")
REGNAME = re.compile(r"[a-z][0-9]+$")


def isa_params(items: list[str]) -> dict:
    """
    --isa-param NAME=VALUE options as a dict
    """
    out = {}
    for item in items or []:
        name, eq, value = item.partition("=")
        if not eq or not name:
            sys.exit(f"--isa-param {item!r}: NAME=VALUE")
        out[name] = value
    return out


def load_experiments(path: str | None) -> dict:
    """
    The experiments: built in, or from a YAML file (name: {about, body,
    repeat})
    """
    if not path:
        return {name: {"about": about, "body": body, "repeat": rep, "pointers": {}}
                for name, (about, body, rep) in BUILTIN.items()}
    with open(path) as f:
        loaded = yaml.safe_load(f) or {}
    out = {}
    for name, e in loaded.items():
        if not isinstance(e, dict) or "body" not in e:
            sys.exit(f"experiment {name!r}: needs a body (a list of instructions)")
        body = e["body"] if isinstance(e["body"], list) else [e["body"]]
        out[str(name)] = {"about": e.get("about", ""), "body": body, "repeat": int(e.get("repeat", 1)),
                          "pointers": dict(e.get("pointers") or {})}
    return out


def evaluate(expr: str, env: dict):
    """
    An expression in an experiment, of i, C, S, T and VB
    """
    try:
        return eval(expr, {"__builtins__": {}}, env)
    except NameError as err:
        if "VB" in str(err):
            sys.exit(f"{expr!r} needs a vector's bytes: --vector-bytes (VLEN/8 - 32 on the K1, 128 "
                     f"on the K3's A100 cores)")
        sys.exit(f"{expr!r}: {err}")


def expand(text: str, env: dict) -> str:
    return TEMPLATE.sub(lambda m: str(evaluate(m.group(1), env)), text)


def instructions(exp: dict, room: dict) -> list[list[str]]:
    """
    An experiment's loop body: its templates expanded per repetition into
    [kind, operands...]
    """
    out = []
    for i in range(exp["repeat"]):
        env = dict(room, i=i)
        for entry in exp["body"]:
            if isinstance(entry, str):
                op = entry
            else:
                op = entry["op"]
                every = int(entry.get("every", 1))
                if i % every != every - 1:
                    continue
                if "when" in entry and not evaluate(str(entry["when"]), env):
                    continue
            text = expand(op, env)
            words = text.split()
            if not words or words[0] not in KINDS or len(words) - 1 not in KINDS[words[0]]:
                sys.exit(f"{text!r}: an instruction is one of "
                         + ", ".join(f"{k} ({' or '.join(map(str, n))} operands)" for k, n in KINDS.items()))
            out.append(words)
    return out


def pointer_offsets(exp: dict, room: dict, label: str) -> dict:
    """
    The experiment's pointers: their offsets into the buffer, checked
    """
    out = {}
    for name, value in exp.get("pointers", {}).items():
        if not re.match(r"p[0-9]+$", str(name)):
            sys.exit(f"{label}: pointers are named p0, p1, ... (not {name!r})")
        off = int(expand(str(value), room))
        if not 0 <= off <= BUF_BYTES - (1 << 16):
            sys.exit(f"{label}: {name} at {off}: pointers start within the buffer's first "
                     f"{BUF_BYTES - (1 << 16)} bytes (64 KB left for them to move)")
        out[str(name)] = off
    return out


class isa_ops:
    """
    The roles on one ISA, written through its asmgen generator
    """
    def __init__(self, isa: str, dt_name: str, params: dict | None = None):
        from asmgen.registers import asm_data_type
        from ukrgen.flow.isa_map import make_generator
        self.isa = isa
        self.gen = make_generator(isa)
        # the generator's parameters (RVV's LMUL, ZVFH, ...) before anything
        # asks it what it can do
        known = list(self.gen.get_parameters() or [])
        for name, value in (params or {}).items():
            if name not in known:
                sys.exit(f"{isa}: no parameter {name} (there are {', '.join(known) or 'none'})")
            self.gen.set_parameter(name, value)
        self.dt = getattr(asm_data_type, dt_name)
        self.ebytes = {"FP64": 8, "FP32": 4, "FP16": 2, "BF16": 2}.get(dt_name, 8)

        def value(attr):
            v = getattr(self.gen, attr)
            return v() if callable(v) else v
        self.nvregs = value("max_vregs")
        self.scalar_tag = "vreg" if value("are_fregs_in_vregs") else "freg"
        big = self.nvregs >= 32
        self.room = {"C": 16 if big else 8, "S": 8 if big else 4, "T": 4 if big else 2}
        if isa in FIXED_VB:
            self.room["VB"] = FIXED_VB[isa]
        self.mask = self.gen.mreg(0) if hasattr(self.gen, "mreg") and "sve" in isa else None
        self.fma_form = self._find_fma_form()

    def _try(self, call):
        try:
            return call()
        except Exception:  # pylint: disable=broad-except
            return None

    def _find_fma_form(self) -> dict | None:
        """
        How this ISA writes an FMA with a scalar operand: the operand
        modifiers and arguments that work, None for none (S a vector)
        """
        from asmgen.asmblocks.op.operand import operand_modifier as om
        g, d = self.gen, self.dt
        s = g.freg(2, d) if "freg" == self.scalar_tag else g.vreg(2)
        forms = []
        if "freg" == self.scalar_tag:
            forms.append({"operand_modifiers": {"bdreg": {om.VF}}})
        # SVE's lane counts within each 128-bit block: its size in elements
        forms += [{"operand_modifiers": {"bdreg": {om.ILANE}}, "bdreg_lane": 0},
                  {"operand_modifiers": {"bdreg": {om.BLOCKLANE}}, "bdreg_lane": 0,
                   "bdreg_blocksize": 16 // self.ebytes}]
        for form in forms:
            if self._try(lambda f=form: g.fma(adreg=g.vreg(1), bdreg=s, cdreg=g.vreg(3),
                                               a_dt=d, b_dt=d, c_dt=d, **f)):
                return form
        return None

    def _masked(self) -> dict:
        from asmgen.asmblocks.op.opd3 import opd3_modifier
        return {"modifiers": {opd3_modifier.MASK}, "amreg": self.mask} if self.mask is not None else {}

    def fma_vv(self, c, a, b) -> str:
        g, d = self.gen, self.dt
        out = self._try(lambda: g.fma(adreg=a, bdreg=b, cdreg=c, a_dt=d, b_dt=d, c_dt=d))
        if out is None and self.mask is not None:
            out = g.fma(adreg=a, bdreg=b, cdreg=c, a_dt=d, b_dt=d, c_dt=d, **self._masked())
        if out is None:
            sys.exit(f"{self.isa}: no vector FMA through asmgen")
        return out

    def fma_s(self, c, a, s) -> str:
        if self.fma_form is None:
            return self.fma_vv(c, a, s)
        d = self.dt
        return self.gen.fma(adreg=a, bdreg=s, cdreg=c, a_dt=d, b_dt=d, c_dt=d, **self.fma_form)

    def vload(self, v, areg) -> str:
        from asmgen.asmblocks.op.opmem import opmem_modifier
        g = self.gen
        out = self._try(lambda: g.load(dregs=[v], areg=areg, dt=self.dt))
        if out is None and self.mask is not None:
            out = g.load(dregs=[v], areg=areg, dt=self.dt, modifiers={opmem_modifier.MASK},
                         amreg=self.mask)
        if out is None:
            sys.exit(f"{self.isa}: no vector load through asmgen")
        return out

    def add_imm(self, reg, imm: int) -> str:
        """
        reg += imm in steps every ISA encodes (RISC-V's addi: -2048..2047)
        """
        out = ""
        while imm > 2047:
            out += self.gen.add_greg_imm(reg=reg, imm=2047)
            imm -= 2047
        while imm < -2048:
            out += self.gen.add_greg_imm(reg=reg, imm=-2048)
            imm += 2048
        if imm:
            out += self.gen.add_greg_imm(reg=reg, imm=imm)
        return out

    def sload(self, idx, areg, offset) -> str:
        """
        A scalar operand loaded into register idx of its class: an FP
        register, or a vector register's first element
        """
        from asmgen.asmblocks.op.opmem import opmem_modifier
        f = self.gen.freg(idx, self.dt)
        if offset:
            return self.gen.load(dregs=[f], areg=areg, dt=self.dt,
                                 modifiers={opmem_modifier.IOFFSET}, ioffset=offset)
        return self.gen.load(dregs=[f], areg=areg, dt=self.dt)

    def function(self, sym: str, body: list[list[str]], label: str,
                 pointers: dict | None = None, loop_offset: int | None = None) -> tuple[str, str]:
        """
        The experiment as a function sym(long n, double *buf): its whole text
        and its loop alone
        """
        from asmgen.callconv.fngen import fngen
        g = self.gen
        cc = g.create_callconv("default")
        cc.add_param("greg", "n")
        cc.add_param("greg", "buf")
        rt = g.get_reg_tracker()
        fg = fngen(g, rt)
        fg.init_cc(cc)
        n = g.greg(rt.aliased_regs["greg"]["n"])
        buf = g.greg(rt.aliased_regs["greg"]["buf"])
        quirks = g.isaquirks(rt=rt, dt=self.dt)
        regs = {}

        def reg(name):
            if not REGNAME.match(name):
                sys.exit(f"{sym}: {name!r} isn't a register name (a class letter and a number)")
            if name not in regs:
                tag = {"c": "vreg", "a": "vreg", "b": "vreg", "v": "vreg",
                       "s": self.scalar_tag, "t": self.scalar_tag, "g": "greg", "p": "greg"}.get(name[0])
                if tag is None:
                    sys.exit(f"{label}: {name!r}: the classes are c a b v (vector), s t (scalar), "
                             f"g (general), p (pointers)")
                try:
                    idx = rt.reserve_any_reg(tag)
                except Exception:  # pylint: disable=broad-except
                    sys.exit(f"{label}: needs more {tag} registers than {self.isa} has free "
                             f"(at {name}, after {len(regs)} names)")
                regs[name] = (tag, idx)
            tag, idx = regs[name]
            if "greg" == tag:
                return g.greg(idx)
            if "freg" == tag:
                return g.freg(idx, self.dt)
            return g.vreg(idx)

        # allocated by class, scalar operands first: an indexed FMA's operand
        # has the lowest registers (SVE's FP64 fmla by lane takes z0-z15)
        names = {w for words in body for w in words[1:] if REGNAME.match(w)}
        pointers = pointers or {}
        for name in pointers:
            names.add(name)
        for cls in "stabvcgp":
            for name in sorted((x for x in names if x[0] == cls), key=lambda x: int(x[1:])):
                reg(name)
        loop = ""
        for words in body:
            kind, ops = words[0], words[1:]
            if "fma" == kind:
                loop += self.fma_s(reg(ops[0]), reg(ops[1]), reg(ops[2]))
            elif "vfma" == kind:
                loop += self.fma_vv(reg(ops[0]), reg(ops[1]), reg(ops[2]))
            elif "vload" == kind:
                loop += self.vload(reg(ops[0]), reg(ops[1]) if len(ops) > 1 else buf)
            elif "sload" == kind:
                reg(ops[0])
                loop += self.sload(regs[ops[0]][1], reg(ops[2]) if len(ops) > 2 else buf, int(ops[1]))
            elif "iadd" == kind:
                loop += self.add_imm(reg(ops[0]), int(ops[1]))
        init = ""
        for name, (tag, idx) in regs.items():
            if name[0] == "c":
                init += g.zero_vreg(vreg=g.vreg(idx), dt=self.dt)
            elif name[0] in "abv":
                init += self.vload(g.vreg(idx), buf)
            elif name[0] in "st":
                init += self.sload(idx, buf, 0)
            elif name[0] == "g":
                init += g.mov_greg_imm(reg=g.greg(idx), imm=0)
            elif name[0] == "p":
                # buffer + offset: a multiple of 64 KB as one immediate (one
                # instruction everywhere, AArch64's mov included), the rest added
                off = pointers.get(name, 0)
                big = off & ~0xFFFF
                if big:
                    init += (g.mov_greg_imm(reg=g.greg(idx), imm=big)
                             + g.add_greg_greg(dst=g.greg(idx), reg1=g.greg(idx), reg2=buf))
                else:
                    init += g.mov_greg(src=buf, dst=g.greg(idx))
                init += self.add_imm(g.greg(idx), off - big)
        lbl = f"{sym}_loop"
        loop = g.loopbegin(reg=n, label=lbl) + loop + g.loopend(reg=n, label=lbl)
        end = g.isaendquirks(rt=rt, dt=self.dt)
        save, load, restore = fg.get_boilerplate(cc)
        # the loop's head on a 16-byte boundary, or loop_offset bytes past a
        # 64-byte one (nops: asmgen's align)
        head = g.align(16) if loop_offset is None else g.align(64, loop_offset)
        text = fg.function(sym, save + load + quirks + init + head + loop + end + restore + g.ret(),
                           align=16)
        return text, loop


DRIVER_MEASURE = r"""
typedef void (*pairs_fn)(long, double*);
static double buf[BUF_DOUBLES] __attribute__((aligned(4096)));
static double span(pairs_fn f, long n, int reps)
{
    double t = 1e300;
    for (int r = 0; r < reps; ++r) {
        double t0 = now(); TIC();
        f(n, buf);
        double d = TOC(t0);
        if (d < t) t = d;
    }
    return t;
}
/* per iteration: 2n iterations less n, the call's cost gone */
static void measure(const char *name, pairs_fn f, long n, int reps)
{
    IN_CHILD(name, {
        span(f, n / 4 + 1, 2);
        double t1 = span(f, n, reps), t2 = span(f, 2 * n, reps);
        printf("pair %s %.4f\n", name, (t2 - t1) / n);
    });
}
"""


def driver(prologue: str, names: list[str]) -> str:
    calls = "".join(f'    measure("{name}", pairs_{i}, n, reps);\n' for i, name in enumerate(names))
    externs = "".join(f"extern void pairs_{i}(long, double*);\n" for i in range(len(names)))
    return (C_HEADER + prologue + externs + f"#define BUF_DOUBLES {BUF_BYTES // 8}\n"
            + DRIVER_MEASURE + C_MAIN_BEGIN
            + "    long n = argc > 1 ? atol(argv[1]) : 2000;\n"
            + "    int reps = argc > 2 ? atoi(argv[2]) : 10;\n"
            + "    for (long i = 0; i < BUF_DOUBLES; ++i) buf[i] = 1e-6;\n"
            + calls + C_MAIN_END)


def parse(outputs: dict) -> tuple[dict, str]:
    """
    Per core per experiment the median of its runs' values; the unit
    """
    from statistics import median
    values, unit = {}, "?"
    for cpu, text in outputs.items():
        seen = {}
        for line in text.splitlines():
            w = line.split()
            if len(w) >= 3 and "#" == w[0] and "unit" == w[1]:
                unit = w[2]
            elif 3 == len(w) and "pair" == w[0]:
                seen.setdefault(w[1], []).append(float(w[2]))
            elif len(w) >= 2 and "crashed" == w[0]:
                seen.setdefault(w[1], []).append(float("nan"))
        values[cpu] = {k: median(v) for k, v in seen.items()}
    return values, unit


def main():
    p = argparse.ArgumentParser(
        prog="ukropt-pairs",
        description="Microbenchmarks of instruction mixes through asmgen: the same experiments on "
                    "every ISA. Without --experiments, built-in ones: FMA throughput by sources, "
                    "FMA latency (chains), FMAs with scalar and vector loads at several ratios.")
    p.add_argument("isa", help="an ISA as ukrgen names it: rvv, neon, sve, avx256, avx512, ...")
    p.add_argument("--experiments", default=None,
                   help="a YAML file of experiments (name: {about, body, repeat}) instead of the "
                        "built-in ones")
    p.add_argument("--only", default=None,
                   help="the experiments to run, names or patterns, comma-separated (fma*,chain-*)")
    p.add_argument("--list", action="store_true", help="list the experiments and stop")
    p.add_argument("--show-asm", action="store_true", help="print each experiment's loop")
    p.add_argument("--dt", default="FP64", help="the data type (asmgen's name): FP64, FP32, ...")
    p.add_argument("--isa-param", action="append", default=[], metavar="NAME=VALUE",
                   help="a generator parameter (RVV: LMUL=2, ZVFH=on), repeatable")
    p.add_argument("--loop-offset", default=None,
                   help="bytes past a 64-byte boundary the loops start at, several "
                        "comma-separated (0,4,8,12): each experiment once per offset - whether "
                        "instructions sharing an aligned 8 or 16 bytes issue together. On "
                        "RISC-V, -march without c keeps every instruction 4 bytes")
    p.add_argument("--vector-bytes", type=int, default=None,
                   help="a vector's bytes (VLEN/8) for experiments using VB, on RVV and SVE")
    p.add_argument("--iterations", type=int, default=2000,
                   help="loop iterations timed (and twice as many: the difference counts)")
    p.add_argument("--measurements", type=int, default=10, help="timings each, the least kept")
    add_common_args(p)
    args = p.parse_args()

    experiments = load_experiments(args.experiments)
    if args.only:
        pats = [x for x in args.only.split(",") if x]
        experiments = {k: v for k, v in experiments.items() if any(fnmatch.fnmatch(k, x) for x in pats)}
        if not experiments:
            sys.exit(f"no experiment matches {args.only!r}")
    if args.list:
        for name, e in experiments.items():
            print(f"{name:16} {e['about']}")
        return

    ops = isa_ops(args.isa, args.dt, isa_params(args.isa_param))
    if args.vector_bytes:
        ops.room["VB"] = args.vector_bytes
    out = args.out or f"ukrpairs-{args.isa}"
    os.makedirs(out, exist_ok=True)
    form = ("a scalar-operand form" if ops.fma_form else "vector FMAs (no scalar form: S a vector)")
    print(f"# {args.isa}: {ops.nvregs} vector registers, room C={ops.room['C']} S={ops.room['S']} "
          f"T={ops.room['T']}; fma as {form}", file=sys.stderr)
    names, texts, facts, abouts = [], [], {}, {}
    offsets = [None] if args.loop_offset is None else \
        [int(x) for x in str(args.loop_offset).split(",") if x.strip()]
    march = re.search(r"-march=rv(?:32|64)([a-z]*)", args.cflags or "")
    if args.isa.startswith("rv") and any(o is not None and o % 4 for o in offsets) \
            and march and "c" not in march.group(1):
        sys.exit(f"--loop-offset {args.loop_offset}: an offset of 2 mod 4 needs 2-byte nops, "
                 f"the C extension - -march=rv64gcv, or offsets that are multiples of 4")
    variants = [(f"{name}@{off}" if off is not None and len(offsets) > 1 else name, name, e, off)
                for name, e in experiments.items() for off in offsets]
    for i, (name, base, e, off) in enumerate(variants):
        body = instructions(e, ops.room)
        text, loop = ops.function(f"pairs_{i}", body, base, pointer_offsets(e, ops.room, base), off)
        names.append(name)
        texts.append(text)
        facts[name] = {"instrs": len(body), "fmas": sum(w[0] in ("fma", "vfma") for w in body)}
        if args.show_asm:
            print(f"# {name}: {e['about']}\n{loop}")
        abouts[name] = e["about"] + (f" (loop at 64 + {off})" if off is not None else "")
    asm = os.path.join(out, "pairs.S")
    with open(asm, "w") as f:
        f.write("\n".join(texts) + ops.gen.asmfile_annotations())
    exe = build(args, out, driver(load_prologue(args.use_prologue), names), [asm], "ukrpairs_bench")
    argv = [str(args.iterations), str(args.measurements)]
    if args.build_only:
        print(f"built {exe}: run '{exe} {' '.join(argv)}' on the target")
        return
    values, unit = parse(run(args, exe, argv))
    cpus = list(values)
    head = [f"core {c}" if c is not None else unit for c in cpus]
    per = "cycle" if "cycles" == unit else unit
    print(f"{unit} per iteration of each loop; {unit} per FMA and instructions per {per}"
          f"{' (IPC)' if 'cycles' == unit else ''} from the least over the cores")
    wide = max([16, *map(len, names)])
    print(" | ".join([f"{'experiment':{wide}}", *[f"{h:>8}" for h in head],
                      f"{'instr':>5}", f"{'per FMA':>7}", f"{'IPC':>5}", "what it shows"]))
    for name in names:
        per = [values[c].get(name, float("nan")) for c in cpus]
        good = [v for v in per if v == v]
        least = min(good) if good else float("nan")
        n_i, n_f = facts[name]["instrs"], facts[name]["fmas"]
        per_fma = least / n_f if n_f and least == least else float("nan")
        ipc = n_i / least if least and least == least else float("nan")
        cells = [f"{v:8.2f}" if v == v else f"{'crashed':>8}" for v in per]
        print(" | ".join([f"{name:{wide}}", *cells, f"{n_i:5d}",
                          f"{per_fma:7.2f}" if per_fma == per_fma else f"{'-':>7}",
                          f"{ipc:5.2f}" if ipc == ipc else f"{'-':>5}", abouts[name]]))


if __name__ == "__main__":
    main()

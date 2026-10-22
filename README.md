# ukropt

Scripts for tuning [ukrgen](https://github.com/FZJ-JSC/jukefa-ukrgen)'s
settings by measuring the microkernels it generates. ukrgen's settings
(issue model, distances, buckets, order, unroll, registers, ...) model a core
only roughly; these scripts measure what they produce instead.

## Installing

ukropt uses ukrgen and asmgen as installed in the current environment - they
are not declared as dependencies (PyPI's `asmgen` is an unrelated package):

    pip install -e /path/to/asmgen
    pip install -e /path/to/jukefa-ukrgen
    pip install -e /path/to/ukropt

## ukropt-sweep - a family's settings searched over a grid

Each combination of a grid's values (every one, or `--sample N`) builds a
kernel of one family and size with blisgen's family builder; all variants go
into one benchmark binary, which times each through its BLIS wrapper with A
and B packed in L1 at several values of k (`--k 16,32,48,64,80`, a
least-squares line through them): cycles per k step, cycles per call outside
the loop, the worst residual (a bent line says the panels left L1). The
variants are ranked and the best printed as YAML for the family file.

    ukropt-sweep ukrconfig/rvv/fma_vf/dgemm.yaml --family vm --size 2x14 \
        --grid examples/grid-k1-vm.yaml --sample 40 \
        --remote k1:/tmp --cc riscv64-linux-gnu-gcc \
        --cflags "-O2 -march=rv64gcv -static"

The grid (YAML) maps family keys or flow parameters to lists of values;
`null` removes the option:

    sched-cycle-slots: ["2:vl|vs|vf,sl,i", "2:vf,vl|vs|vf,sl,sl,i;vf-sl"]
    schedule: [["RAW:data::::13"], ["RAW:data:freg:::6", "RAW:data:vreg:::12"]]
    buckets: [null, "vf:3", "vf:5"]
    order: [null, knm]
    unroll: [2, 4]

Options:

- `-j N` - variants generated in parallel (all cores by default).
- `--align 0,4,8,12,16,32` - each variant built at each byte offset from a
  64-byte boundary; its best and worst reported. A difference between
  variants smaller than their own spread is placement, not settings.
- `--remote HOST:DIR` - generate and cross-compile here, copy the one binary
  there (scp), run it (ssh), rank here. With a cross compiler, `-static`
  saves a sysroot.
- `--build-only` - build the binary and stop; run `bench k1 k2 ...` on the
  target yourself (`variants.yaml` maps variant numbers to settings).
- `--perf-lib DIR` - count with
  [performance_counters](https://github.com/linedot/performance_counters)
  (`--perf-backend perf|papi|m1|simple`, `--event`, `--cxx`) instead of the
  built-in perf_event_open timer (wall-clock where counters aren't there).

- `--cpu N` or `--cpu 1,5` - measure on those cores: the benchmark pins
  itself (sched_setaffinity, inherited by the variants' processes), once
  per core. On the K1 the two clusters differ by a few cycles a k step, in
  either direction depending on the kernel.
- `--rank mean|max|min|N` - with several cores, rank by the mean (the
  default), the worst core (`max`: a kernel good everywhere), the best, or
  one core; the table has a column per core.
- `--use-prologue NAME` - prepare the core before anything else runs:
  `src/ukropt/prologues/NAME.h`, or a header's path. A prologue defines
  `UKRSWEEP_PROLOGUE(argc, argv)`, which the benchmark's main runs first.
  `k3_ai`: the SpacemiT K3's AI cores (8-15) - the benchmark registers its
  PID with `/proc/set_ai_thread` and re-executes itself, so nothing has
  used the vector unit before (as https://github.com/brucehoult/k3_ai does);
  then `--cpu 8` and on.
- `--measurements N` (default 20; formerly `--reps`) - timed measurements
  per variant and k in one run, the fastest kept.
- `--iterations N` (default 64) - kernel calls back to back in one
  measurement.
- `--runs N` (default 1; formerly `--repeat`) - the whole benchmark run N
  times, each point the median of the runs' fastest.
- `--max-residual P` (default 2) - fits whose worst residual exceeds P
  percent are marked `?` and not chosen best: their points aren't a line.

Each variant runs in a child process of its own: one that crashes is
reported (variant, offset, signal) and the others still run.

The BLIS declarations the wrappers compile against come from ukrgen where it
ships them, else from the copy in `src/ukropt/data/blis_stub.h`.

## ukropt-pairs - how a core executes instruction mixes

Microbenchmarks generated through asmgen: an experiment is written once, in
ISA-neutral roles, and runs on every ISA asmgen has (RVV, NEON, SVE, AVX2,
AVX-512, ...). The built-in experiments measure FMA throughput by the number
of vector sources, FMA latency (dependency chains), and FMAs mixed with
scalar loads, vector loads and integer adds at several ratios:

```
ukropt-pairs rvv --cc riscv64-linux-gnu-gcc --cflags "-O2 -march=rv64gcv -static" \
    --remote host:/tmp --cpu 8,11 --use-prologue k3_ai
ukropt-pairs avx512 --only "fma*,chain-*"          # native
ukropt-pairs neon --list                           # the experiments
```

It reports per experiment the cycles per loop iteration on each core, cycles
per FMA and instructions per cycle. `--show-asm` prints each loop.

Experiments of your own go in a YAML file (`--experiments`); see
`examples/pairs-gemm-mixes.yaml`, the instruction mixes of GEMM microkernels
without dependences - a kernel's cycles per k step against its mix's:

```yaml
mix-2vx14:
  about: "28 FMAs, 14 scalar loads, 2 vector loads"
  repeat: 14                       # the body expanded for i = 0..13
  body:
    - "fma c{(2*i)%C} a0 s{i%S}"   # C += A * S, S a scalar operand
    - "fma c{(2*i+1)%C} a1 s{i%S}"
    - "sload t{i%T} {8*i}"         # a scalar loaded, OFFSET bytes in
    - {op: "vload v{i%2}", every: 7}   # only where i % 7 == 6
```

The roles: `fma C A S` (RVV vfmacc.vf, NEON/SVE fmla by element; x86 has no
scalar form, S is a vector there), `vfma C A B`, `vload V`, `sload S OFFSET`,
`iadd G IMM`. Register names give their class: `c` accumulators, `a b`
vector sources, `v` vector load targets, `s t` scalar operands, `g` general
registers; each name is a register of its own, allocated through asmgen. An
experiment needing more registers than the ISA has is reported. `{...}` is
an expression of `i` and of `C S T`, the accumulators and scalar operands
there is room for (16, 8, 4 with 32 vector registers; 8, 4, 2 with 16).

Each loop runs n and 2n iterations; the difference per iteration counts, the
call's cost gone. The building, running and counting options are
ukropt-sweep's (`--cpu`, `--runs`, `--use-prologue`, `--remote`,
`--perf-lib`), shared through `ukropt/harness.py`.

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

Each variant runs in a child process of its own: one that crashes is
reported (variant, offset, signal) and the others still run.

The BLIS declarations the wrappers compile against come from ukrgen where it
ships them, else from the copy in `src/ukropt/data/blis_stub.h`.

"""
What ukropt's benchmarks share: the C timing code (perf_event cycles, or
wall-clock, or linedot/performance_counters), pinning to a core, prologues
preparing a core, a child process per measurement (a crash reported, the
rest run), building - natively or cross - and running, here or on a remote
host, once per core.

A benchmark's driver includes C_HEADER, defines its measurements and calls
them from main through C_MAIN_BEGIN / C_MAIN_END.
"""

import argparse
import os
import shutil
import subprocess
import sys

C_HEADER = r"""
#define _GNU_SOURCE
#include <sched.h>
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
#define OPEN_COUNTER() pc_init(PC_EVENT)
#else
#define OPEN_COUNTER() counter_open()
#endif
/* a timestamp: cycles where the counter opened, else nanoseconds */
static double now(void)
{
    if (cycles_fd >= 0) { uint64_t c = 0; if (read(cycles_fd, &c, sizeof c) == sizeof c) return (double)c; }
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec;
}
#ifdef USE_PCLIB
#define TIC() pc_tic()
#define TOC(t0) ((double)pc_toc())
#else
#define TIC() ((void)0)
#define TOC(t0) (now() - (t0))
#endif
#ifndef UKRSWEEP_PROLOGUE
#define UKRSWEEP_PROLOGUE(argc, argv) ((void)0)
#endif
/* a measurement in a child of its own: a crash reported, the others run */
#define IN_CHILD(label, ...) do { pid_t p_ = fork(); if (0 == p_) { OPEN_COUNTER(); __VA_ARGS__; \
    fflush(stdout); _exit(0); } int st_ = 0; waitpid(p_, &st_, 0); \
    if (!WIFEXITED(st_) || WEXITSTATUS(st_)) { printf("crashed %s %d\n", label, \
    WIFSIGNALED(st_) ? WTERMSIG(st_) : WEXITSTATUS(st_)); fflush(stdout); } } while (0)
"""

C_MAIN_BEGIN = r"""
int main(int argc, char** argv)
{
    /* a core's preparation, first of all (--use-prologue) */
    UKRSWEEP_PROLOGUE(argc, argv);
    /* pinned where asked (UKRSWEEP_CPU): the children inherit it */
    const char *cpu = getenv("UKRSWEEP_CPU");
    if (cpu && *cpu) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(atoi(cpu), &set);
        if (sched_setaffinity(0, sizeof set, &set))
            perror("sched_setaffinity");
    }
#ifdef USE_PCLIB
    printf("# unit %s\n", PC_EVENT);
#else
    counter_open();
    if (cycles_fd >= 0) { printf("# unit cycles\n"); close(cycles_fd); cycles_fd = -1; }
    else printf("# unit ns\n");
#endif
    fflush(stdout);
"""

C_MAIN_END = r"""
    return 0;
}
"""


def load_prologue(name: str | None) -> str:
    """
    A prologue for a benchmark (--use-prologue): ukropt/prologues/NAME.h,
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


def add_common_args(p: argparse.ArgumentParser):
    """
    The options every ukropt benchmark takes: building, where to run, how to
    measure
    """
    p.add_argument("--out", default=None, help="where to build (default: a directory per tool)")
    p.add_argument("--cc", default="cc", help="C compiler (a cross compiler with --remote)")
    p.add_argument("--cflags", default="-O2", help="compiler flags")
    p.add_argument("--remote", default=None,
                   help="HOST:DIR - run there (scp, ssh), build here; with a cross compiler, "
                        "-static saves a sysroot")
    p.add_argument("--build-only", action="store_true", help="build the binary and stop")
    p.add_argument("--cpu", default=None,
                   help="the core(s) to measure on, 1 or 1,5: the benchmark pins itself")
    p.add_argument("--runs", type=int, default=1,
                   help="the whole benchmark run this many times, each value the median")
    p.add_argument("--use-prologue", default=None,
                   help="prepare the core first: a name in ukropt/prologues (k3_ai) or a "
                        "header's path defining UKRSWEEP_PROLOGUE(argc, argv)")
    p.add_argument("--perf-lib", default=None,
                   help="count with linedot/performance_counters at this path instead")
    p.add_argument("--perf-backend", default="perf", help="performance_counters backend")
    p.add_argument("--event", default="cycles", help="the event to count with --perf-lib")
    p.add_argument("--cxx", default="c++", help="C++ compiler for --perf-lib")


def build(args, out: str, driver: str, sources: list[str], name: str = "bench") -> str:
    """
    The benchmark built from its driver's C text and its assembly sources:
    the executable's path
    """
    os.makedirs(out, exist_ok=True)
    driver_c = os.path.join(out, "driver.c")
    with open(driver_c, "w") as f:
        f.write(driver)
    exe = os.path.join(out, name)
    if args.perf_lib:
        # performance_counters (C++) behind three C functions
        shim = os.path.join(out, "pc_shim.cpp")
        with open(shim, "w") as f:
            f.write('#include "performance_counters.hpp"\n'
                    'static performance_counters* pc;\n'
                    'extern "C" int pc_init(const char* e) { pc = new performance_counters('
                    'std::vector<std::string>{e}, false); return 1; }\n'
                    'extern "C" void pc_tic() { pc->tic(); }\n'
                    'extern "C" unsigned long long pc_toc() { return pc->toc()[0]; }\n')
        steps = [[args.cc, *args.cflags.split(), "-DUSE_PCLIB", f'-DPC_EVENT="{args.event}"', "-c",
                  driver_c, "-o", os.path.join(out, "driver.o")],
                 [args.cxx, *args.cflags.split(), "-std=c++17", f"-I{args.perf_lib}", "-o", exe,
                  os.path.join(out, "driver.o"), shim,
                  os.path.join(args.perf_lib, f"performance_counters_{args.perf_backend}.cpp"), *sources]
                 + (["-lpapi"] if "papi" == args.perf_backend else [])]
    else:
        steps = [[args.cc, *args.cflags.split(), "-o", exe, driver_c, *sources]]
    for step in steps:
        r = subprocess.run(step, capture_output=True, text=True)
        if r.returncode:
            sys.exit(r.stderr[:3000])
    return exe


def cpus_of(args) -> list:
    """
    The cores to measure on: [None] where not given
    """
    if args.cpu is None:
        return [None]
    return [int(c) for c in str(args.cpu).split(",") if c.strip()]


def run(args, exe: str, argv: list[str]) -> dict:
    """
    The benchmark run on each core (pinned), --runs times, here or on
    --remote: per core its outputs joined
    """
    if args.remote:
        host, _, where = args.remote.partition(":")
        where = where or "."
        remote_exe = f"{where}/{os.path.basename(exe)}"
        r = subprocess.run(["scp", exe, f"{host}:{remote_exe}"], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"scp failed: {r.stderr}")
    runs = {}
    for cpu in cpus_of(args):
        outs = []
        for _ in range(max(1, args.runs)):
            if args.remote:
                pin = f"UKRSWEEP_CPU={cpu} " if cpu is not None else ""
                r = subprocess.run(["ssh", host, f"cd {where} && {pin}./{os.path.basename(exe)} "
                                    + " ".join(argv)], capture_output=True, text=True, timeout=3600)
            else:
                env = dict(os.environ, **({"UKRSWEEP_CPU": str(cpu)} if cpu is not None else {}))
                r = subprocess.run([exe, *argv], capture_output=True, text=True, timeout=3600, env=env)
            if r.returncode:
                print(f"run (core {cpu}) ended with status {r.returncode}: "
                      f"{r.stderr.strip() or '(nothing on stderr)'} - the results it printed follow",
                      file=sys.stderr)
            outs.append(r.stdout)
        runs[cpu] = "\n".join(outs)
    return runs

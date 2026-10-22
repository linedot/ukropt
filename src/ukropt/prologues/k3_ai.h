/*
 * k3_ai: run the benchmark on the SpacemiT K3's A100 "AI" cores (8-15).
 *
 * A process runs there once its PID is written to /proc/set_ai_thread. As
 * https://github.com/brucehoult/k3_ai does, nothing may have looked at the
 * vector unit before (libc's startup decides on the other cores' VLEN), so
 * the benchmark registers itself and re-executes its own binary: the PID -
 * and the registration - survive execve, the new image starts registered.
 * Forked children (the variants' processes) inherit the registration.
 */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static void ukrsweep_prologue(int argc, char **argv)
{
    (void)argc;
    if (getenv("UKRSWEEP_K3_AI"))
        return;
    char pid[32];
    int n = snprintf(pid, sizeof pid, "%d", (int)getpid());
    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd < 0 || write(fd, pid, n) != n) {
        perror("k3_ai: /proc/set_ai_thread");
        exit(1);
    }
    close(fd);
    setenv("UKRSWEEP_K3_AI", "1", 1);
    execv("/proc/self/exe", argv);
    perror("k3_ai: execv /proc/self/exe");
    exit(1);
}

#define UKRSWEEP_PROLOGUE(argc, argv) ukrsweep_prologue(argc, argv)

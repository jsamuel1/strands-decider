// E14: does setitimer(ITIMER_PROF)'s SIGPROF land on the thread that used the CPU, or mostly on the main thread?
//
//   cc -O2 -pthread sigprof_bias.c -o sigprof_bias && ./sigprof_bias [threads=6] [seconds=3]
//
// lkaisamp (profiling/lkaisamp.c) arms setitimer(ITIMER_PROF) and records the thread the kernel delivered SIGPROF to.
// This program reproduces exactly that: `threads` threads (the main thread included) spin with equal work, and it
// prints the share of signals each received. With fair delivery every thread gets 1/threads. Kernel releases differ
// here (Linux 6.4 changed process-wide timer signals to prefer the current thread; older kernels send them to the
// thread-group leader unless it blocks them or is not running), so the result depends on the kernel it runs on.
// It prints the kernel release beside the shares. Standard C and POSIX only.
#define _GNU_SOURCE
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

#define MAXT 64
static _Atomic long hits[MAXT];
static __thread int my_idx = -1;
static _Atomic int stop;

static void on_prof(int sig) {
    (void) sig;
    if (my_idx >= 0) atomic_fetch_add_explicit(&hits[my_idx], 1, memory_order_relaxed);
}

static void spin(void) {
    volatile unsigned long x = 0;
    while (!atomic_load_explicit(&stop, memory_order_relaxed)) x++;
}

static void * worker(void * arg) {
    my_idx = (int) (long) arg;
    spin();
    return NULL;
}

int main(int argc, char ** argv) {
    int n = argc > 1 ? atoi(argv[1]) : 6;
    int secs = argc > 2 ? atoi(argv[2]) : 3;
    if (n < 1 || n > MAXT) return 2;
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_prof;
    sa.sa_flags = SA_RESTART;
    sigaction(SIGPROF, &sa, NULL);
    pthread_t t[MAXT];
    my_idx = 0;
    for (int i = 1; i < n; i++) pthread_create(&t[i], NULL, worker, (void *) (long) i);
    struct itimerval it = {{0, 1000}, {0, 1000}};  // 1 ms of process CPU time, as lkaisamp's period_us=1000
    setitimer(ITIMER_PROF, &it, NULL);
    struct timespec t0, now;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (;;) {
        for (volatile int k = 0; k < 1000000; k++) {
        }
        clock_gettime(CLOCK_MONOTONIC, &now);
        if (now.tv_sec - t0.tv_sec >= secs) break;
    }
    struct itimerval off = {{0, 0}, {0, 0}};
    setitimer(ITIMER_PROF, &off, NULL);
    atomic_store(&stop, 1);
    for (int i = 1; i < n; i++) pthread_join(t[i], NULL);
    struct utsname u;
    uname(&u);
    long tot = 0;
    for (int i = 0; i < n; i++) tot += hits[i];
    printf("{\"kernel\":\"%s\",\"machine\":\"%s\",\"threads\":%d,\"signals\":%ld,\"share\":[", u.release, u.machine, n, tot);
    for (int i = 0; i < n; i++) printf("%s%.4f", i ? "," : "", tot ? (double) hits[i] / tot : 0.0);
    printf("],\"main_thread_share\":%.4f,\"fair_share\":%.4f}\n", tot ? (double) hits[0] / tot : 0.0, 1.0 / n);
    return 0;
}

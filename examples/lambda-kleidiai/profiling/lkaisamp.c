// lkaisamp: an in-process native CPU sampler for Lambda, where perf, ptrace-based profilers and eBPF may be denied.
//
// setitimer(ITIMER_PROF) raises SIGPROF every `period_us` of process CPU time (user + system, all threads); the kernel
// delivers it to a thread that is consuming CPU. The SA_SIGINFO handler stores that thread's interrupted program counter
// and, on aarch64, its link register (the caller of a leaf function), the current phase tag and the thread id. Nothing
// else runs in the handler: no allocation, no locks, no unwinding. Symbolisation happens offline from /proc/self/maps.
//
// Python drives it through ctypes (profiling/lkai_prof.py):
//   samp_start(period_us, capacity)  allocate the buffer once (before the timer exists) and arm the timer
//   samp_phase(tag)                  tag later samples (0..255); returns the previous tag
//   samp_stop()                      disarm the timer
//   samp_count() / samp_dropped()    samples stored / lost to a full buffer
//   samp_dump(fd)                    write a binary record: header, then n x {pc, lr, tid, tag} (little endian)
//   samp_reset()                     forget the stored samples (keeps the buffer)
//
// Time in the kernel (page faults, memcpy into a memfd, futex waits do not count: they use no CPU) is charged to the
// user instruction that entered it, so a page-fault-heavy forward shows the matmul kernel that touched the page.
#define _GNU_SOURCE
#include <signal.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <ucontext.h>
#include <unistd.h>

typedef struct {
    uint64_t pc;
    uint64_t lr;
    uint32_t tid;
    uint32_t tag;
} samp_t;

static samp_t * buf;
static uint64_t cap;
static volatile uint64_t n_next;   // next slot (may exceed cap: the excess is the dropped count)
static volatile uint32_t cur_tag;
static uint64_t period;

static void on_prof(int sig, siginfo_t * si, void * uc_) {
    (void) sig;
    (void) si;
    ucontext_t * uc = (ucontext_t *) uc_;
    uint64_t i = __atomic_fetch_add(&n_next, 1, __ATOMIC_RELAXED);
    if (i >= cap) {
        return;
    }
    samp_t * s = &buf[i];
#if defined(__aarch64__)
    s->pc = uc->uc_mcontext.pc;
    s->lr = uc->uc_mcontext.regs[30];
#elif defined(__x86_64__)
    s->pc = (uint64_t) uc->uc_mcontext.gregs[REG_RIP];
    s->lr = 0;
#else
    s->pc = 0;
    s->lr = 0;
#endif
    s->tid = (uint32_t) syscall(SYS_gettid);
    s->tag = cur_tag;
}

int samp_start(int period_us, long capacity) {
    if (!buf) {
        cap = capacity > 0 ? (uint64_t) capacity : (1u << 20);
        buf = calloc(cap, sizeof(samp_t));  // touched now, so the handler never faults on a fresh page later
        if (!buf) {
            return -1;
        }
    }
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = on_prof;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGPROF, &sa, NULL) != 0) {
        return -2;
    }
    period = (uint64_t) period_us;
    struct itimerval it = {{period_us / 1000000, period_us % 1000000}, {period_us / 1000000, period_us % 1000000}};
    return setitimer(ITIMER_PROF, &it, NULL) == 0 ? 0 : -3;
}

int samp_stop(void) {
    struct itimerval it;
    memset(&it, 0, sizeof(it));
    return setitimer(ITIMER_PROF, &it, NULL);
}

int samp_phase(int tag) {
    return (int) __atomic_exchange_n(&cur_tag, (uint32_t) tag, __ATOMIC_RELAXED);
}

long samp_count(void) { return (long) (n_next < cap ? n_next : cap); }
long samp_dropped(void) { return (long) (n_next > cap ? n_next - cap : 0); }

void samp_reset(void) { __atomic_store_n(&n_next, 0, __ATOMIC_RELAXED); }

static int write_all(int fd, const void * p, size_t n) {
    const char * c = (const char *) p;
    while (n) {
        ssize_t w = write(fd, c, n);
        if (w <= 0) {
            return -1;
        }
        c += w;
        n -= (size_t) w;
    }
    return 0;
}

// Header: magic "LKSAMP01", u64 period_us, u64 n, u64 dropped, u64 sizeof(samp_t); then n records.
int samp_dump(int fd) {
    uint64_t n = (uint64_t) samp_count();
    uint64_t hdr[4] = {period, n, (uint64_t) samp_dropped(), sizeof(samp_t)};
    if (write_all(fd, "LKSAMP01", 8) || write_all(fd, hdr, sizeof(hdr))) {
        return -1;
    }
    return write_all(fd, buf, n * sizeof(samp_t));
}

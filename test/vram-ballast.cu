// test/vram-ballast.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <errno.h>
#include <limits.h>
#include <unistd.h>

static volatile sig_atomic_t running = 1;

static void handle_signal(int sig) {
    (void)sig;
    running = 0;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <MB>\n", argv[0]);
        return 1;
    }

    // Validate input (reject negative, non-numeric, overflow)
    char *endptr;
    errno = 0;
    unsigned long mb = strtoul(argv[1], &endptr, 10);
    if (errno != 0 || *endptr != '\0' || mb == 0 || mb > 100000) {
        fprintf(stderr, "Invalid MB value: %s (must be 1-100000)\n", argv[1]);
        return 1;
    }

    size_t bytes = mb * 1024UL * 1024UL;

    // Report VRAM state before allocation
    size_t free_before, total;
    cudaMemGetInfo(&free_before, &total);
    printf("VRAM before: %.1f MiB free / %.1f MiB total\n",
           free_before / (1024.0 * 1024.0), total / (1024.0 * 1024.0));

    void *ptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Report VRAM state after allocation
    size_t free_after;
    cudaMemGetInfo(&free_after, &total);
    printf("VRAM after:  %.1f MiB free / %.1f MiB total\n",
           free_after / (1024.0 * 1024.0), total / (1024.0 * 1024.0));
    printf("Allocated %lu MB VRAM. PID %d.\n", mb, getpid());

    // Signal readiness for test scripts
    printf("BALLAST_READY=1\n");
    fflush(stdout);

    // Use sigaction for reliable signal handling
    struct sigaction sa;
    sa.sa_handler = handle_signal;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGHUP, &sa, NULL);

    while (running) { sleep(1); }

    // Note: cudaFree is NOT async-signal-safe, but we're past the signal
    // handler here (running=0 breaks the loop, then we execute sequentially)
    cudaFree(ptr);
    printf("Released VRAM. Exiting.\n");
    return 0;
}

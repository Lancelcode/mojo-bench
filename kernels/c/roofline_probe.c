/*
 * Measures the two roofs of a roofline model on this machine:
 *
 *   peak   double-precision FMA throughput (GFLOP/s), using many independent
 *          512-bit FMA chains per thread so latency is hidden;
 *   bw     memory bandwidth (GB/s) on arrays far larger than the L3 cache:
 *          a STREAM-style triad a = b + s*c (24 bytes per element, the STREAM
 *          convention, which ignores write-allocate) and a read-only sweep.
 *
 * Each measurement is taken with 1 thread and with all threads. Output is
 * one "PROBE <name> <threads> <value>" line per measurement.
 *
 * Build: gcc -O3 -march=native -mprefer-vector-width=512 -fopenmp roofline_probe.c
 */
#define _POSIX_C_SOURCE 199309L
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef double v8d __attribute__((vector_size(64)));

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

#define CHAINS 16
static double peak_gflops(int threads) {
    const long iters = 50000000L;
    double best = 0.0;
    for (int rep = 0; rep < 5; rep++) {
        double sink = 0.0;
        double t0 = now();
#pragma omp parallel num_threads(threads) reduction(+ : sink)
        {
            v8d acc[CHAINS];
            v8d m = {0}, a = {0};
            for (int l = 0; l < 8; l++) {
                m[l] = 0.999999;
                a[l] = 1e-7 * (omp_get_thread_num() + 1);
            }
            for (int c = 0; c < CHAINS; c++)
                for (int l = 0; l < 8; l++) acc[c][l] = (double)(c + l);
            for (long i = 0; i < iters; i++)
                for (int c = 0; c < CHAINS; c++) acc[c] = acc[c] * m + a; /* one FMA */
            for (int c = 0; c < CHAINS; c++)
                for (int l = 0; l < 8; l++) sink += acc[c][l];
        }
        double t = now() - t0;
        double flops = 2.0 * 8 * CHAINS * (double)iters * threads;
        if (flops / t > best) best = flops / t;
        if (sink == 42.0) puts("");  /* keep the result alive */
    }
    return best / 1e9;
}

static void bandwidth(int threads, long n, double *b, double *c, double *a, double *triad, double *read) {
    double best_t = 1e30, best_r = 1e30;
    for (int rep = 0; rep < 10; rep++) {
        double t0 = now();
#pragma omp parallel for schedule(static) num_threads(threads)
        for (long i = 0; i < n; i++) a[i] = b[i] + 3.0 * c[i];
        double t = now() - t0;
        if (t < best_t) best_t = t;

        /* Read-only sweep with four independent vector accumulators, so the
         * loop is limited by loads rather than by floating-point add latency. */
        double s = 0.0;
        t0 = now();
#pragma omp parallel num_threads(threads) reduction(+ : s)
        {
            int nt = omp_get_num_threads(), id = omp_get_thread_num();
            long nv = n / 8, lo = nv * id / nt, hi = nv * (id + 1) / nt;
            const v8d *av = (const v8d *)a;
            v8d s0 = {0}, s1 = {0}, s2 = {0}, s3 = {0};
            long i = lo;
            for (; i + 4 <= hi; i += 4) {
                s0 += av[i];
                s1 += av[i + 1];
                s2 += av[i + 2];
                s3 += av[i + 3];
            }
            for (; i < hi; i++) s0 += av[i];
            v8d tot = s0 + s1 + s2 + s3;
            for (int l = 0; l < 8; l++) s += tot[l];
        }
        t = now() - t0;
        if (t < best_r) best_r = t;
        if (s == 42.0) puts("");
    }
    *triad = 24.0 * (double)n / best_t / 1e9;
    *read = 8.0 * (double)n / best_r / 1e9;
}

int main(void) {
    int all = omp_get_num_procs();
    const long n = 60000000L; /* 3 x 480 MB, far beyond L3 */
    double *a = aligned_alloc(64, sizeof(double) * n);
    double *b = aligned_alloc(64, sizeof(double) * n);
    double *c = aligned_alloc(64, sizeof(double) * n);
    if (!a || !b || !c) {
        fprintf(stderr, "out of memory\n");
        return 2;
    }
    /* First touch in parallel so pages are spread as the kernels use them. */
#pragma omp parallel for schedule(static) num_threads(all)
    for (long i = 0; i < n; i++) {
        a[i] = 0.0;
        b[i] = 1.0 + (double)(i % 7);
        c[i] = 2.0 - (double)(i % 5);
    }
    int counts[2] = {1, all};
    for (int k = 0; k < 2; k++) {
        int t = counts[k];
        double triad, read;
        printf("PROBE peak_gflops %d %.2f\n", t, peak_gflops(t));
        bandwidth(t, n, b, c, a, &triad, &read);
        printf("PROBE triad_gbs %d %.2f\n", t, triad);
        printf("PROBE read_gbs %d %.2f\n", t, read);
        fflush(stdout);
    }
    free(a);
    free(b);
    free(c);
    return 0;
}

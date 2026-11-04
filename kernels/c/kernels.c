/*
 * C implementations of the mojo-bench kernels.
 *
 * Usage (same protocol as every other implementation):
 *   kernels_c <kernel> <n> <warmup> <repeats> [threads]
 * prints
 *   RESULT <kernel> <checksum> <t1,t2,...>
 *
 * Plain, idiomatic C compiled with gcc -O3 -march=native (GCC's default
 * gnu17 dialect, which allows FMA contraction). No intrinsics:
 * any SIMD here comes from the compiler's auto-vectoriser, which is what a
 * typical C programmer actually gets.
 */
#define _POSIX_C_SOURCE 199309L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void *xmalloc(size_t bytes) {
    void *p = aligned_alloc(64, (bytes + 63) / 64 * 64);
    if (!p) {
        fprintf(stderr, "out of memory\n");
        exit(2);
    }
    return p;
}

/* Each kernel provides setup (untimed), run (timed) and checksum (untimed). */
typedef struct {
    long n;
    double *a, *b, *c;               /* matmul */
    double *x, *y, *z, *m;           /* n-body initial state */
    double *px, *py, *pz, *vx, *vy, *vz; /* n-body working state */
    int64_t *ia, *ib;                /* reduction / scan */
    int64_t isum;
    uint64_t *keys, *queries, *tkeys; /* hash table */
    int64_t *tvals;
    long cap;
    int64_t found, vsum, total;
} State;

/* ------------------------------------------------------------- matmul */

static void mat_setup(State *s) {
    long n = s->n;
    s->a = xmalloc(sizeof(double) * n * n);
    s->b = xmalloc(sizeof(double) * n * n);
    s->c = xmalloc(sizeof(double) * n * n);
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++) {
            s->a[i * n + j] = (double)((7 * i + 3 * j) % 11 - 5);
            s->b[i * n + j] = (double)((5 * i + 13 * j) % 7 - 3);
        }
}

static void matmul_naive(State *s) {
    long n = s->n;
    const double *a = s->a, *b = s->b;
    double *c = s->c;
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++) {
            double acc = 0.0;
            for (long k = 0; k < n; k++)
                acc += a[i * n + k] * b[k * n + j];
            c[i * n + j] = acc;
        }
}

#define TILE 64
static void matmul_tiled(State *s) {
    long n = s->n;
    const double *restrict a = s->a, *restrict b = s->b;
    double *restrict c = s->c;
    memset(c, 0, sizeof(double) * n * n);
    for (long ii = 0; ii < n; ii += TILE)
        for (long kk = 0; kk < n; kk += TILE)
            for (long jj = 0; jj < n; jj += TILE) {
                long ie = ii + TILE < n ? ii + TILE : n;
                long ke = kk + TILE < n ? kk + TILE : n;
                long je = jj + TILE < n ? jj + TILE : n;
                for (long i = ii; i < ie; i++)
                    for (long k = kk; k < ke; k++) {
                        double aik = a[i * n + k];
                        for (long j = jj; j < je; j++) /* unit stride: vectorises */
                            c[i * n + j] += aik * b[k * n + j];
                    }
            }
}

static void mat_checksum(State *s, char *out) {
    long n = s->n;
    double acc = 0.0;
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++) {
            double v = s->c[i * n + j];
            acc += v * (double)((3 * i + 5 * j) % 13 + 1) + v * v;
        }
    sprintf(out, "%lld", (long long)llround(acc));
}

/* ------------------------------------------------------------- mandelbrot */

#define MAX_ITER 200
static void mandelbrot(State *s) {
    long n = s->n;
    int64_t total = 0;
    for (long i = 0; i < n; i++) {
        double y0 = -1.25 + 2.5 * (double)i / (double)n;
        for (long j = 0; j < n; j++) {
            double x0 = -2.0 + 2.5 * (double)j / (double)n;
            double x = 0.0, y = 0.0;
            int k = 0;
            while (k < MAX_ITER && x * x + y * y <= 4.0) {
                double xt = x * x - y * y + x0;
                y = 2.0 * x * y + y0;
                x = xt;
                k++;
            }
            total += k;
        }
    }
    s->total = total;
}

/* ------------------------------------------------------------- n-body */

#define STEPS 5
#define DT 0.01
#define EPS2 0.01

static void nbody_setup(State *s) {
    long n = s->n;
    double **arrs[] = {&s->x, &s->y, &s->z, &s->m, &s->px, &s->py,
                       &s->pz, &s->vx, &s->vy, &s->vz};
    for (int t = 0; t < 10; t++) *arrs[t] = xmalloc(sizeof(double) * n);
    for (long i = 0; i < n; i++) {
        s->x[i] = (double)((i * 37) % 101) / 101 - 0.5;
        s->y[i] = (double)((i * 53) % 103) / 103 - 0.5;
        s->z[i] = (double)((i * 71) % 107) / 107 - 0.5;
        s->m[i] = 1.0 + (double)(i % 7) / 7;
    }
}

static void nbody(State *s) {
    long n = s->n;
    double *restrict x = s->px, *restrict y = s->py, *restrict z = s->pz;
    double *restrict vx = s->vx, *restrict vy = s->vy, *restrict vz = s->vz;
    const double *restrict m = s->m;
    memcpy(x, s->x, sizeof(double) * n);
    memcpy(y, s->y, sizeof(double) * n);
    memcpy(z, s->z, sizeof(double) * n);
    memset(vx, 0, sizeof(double) * n);
    memset(vy, 0, sizeof(double) * n);
    memset(vz, 0, sizeof(double) * n);
    for (int step = 0; step < STEPS; step++) {
        for (long i = 0; i < n; i++) {
            double xi = x[i], yi = y[i], zi = z[i];
            double ax = 0.0, ay = 0.0, az = 0.0;
            for (long j = 0; j < n; j++) {
                double dx = x[j] - xi, dy = y[j] - yi, dz = z[j] - zi;
                double d2 = dx * dx + dy * dy + dz * dz + EPS2;
                double sc = m[j] / (d2 * sqrt(d2));
                ax += dx * sc;
                ay += dy * sc;
                az += dz * sc;
            }
            vx[i] += ax * DT;
            vy[i] += ay * DT;
            vz[i] += az * DT;
        }
        for (long i = 0; i < n; i++) {
            x[i] += vx[i] * DT;
            y[i] += vy[i] * DT;
            z[i] += vz[i] * DT;
        }
    }
}

static void nbody_checksum(State *s, char *out) {
    double sx = 0, sy = 0, sz = 0, sv = 0;
    for (long i = 0; i < s->n; i++) {
        sx += s->px[i];
        sy += s->py[i];
        sz += s->pz[i];
        sv += s->vx[i] + s->vy[i] + s->vz[i];
    }
    sprintf(out, "%.17g", sx + 2 * sy + 3 * sz + sv);
}

/* ------------------------------------------------------------- reduction / scan */

static void int_setup(State *s) {
    s->ia = xmalloc(sizeof(int64_t) * s->n);
    s->ib = xmalloc(sizeof(int64_t) * s->n);
    for (long i = 0; i < s->n; i++) s->ia[i] = ((int64_t)i * 2654435761LL) % 1000003;
}

static void reduction(State *s) {
    const int64_t *a = s->ia;
    int64_t acc = 0;
    for (long i = 0; i < s->n; i++) acc += a[i];
    s->isum = acc;
}

static void prefix_sum(State *s) {
    const int64_t *restrict a = s->ia;
    int64_t *restrict p = s->ib;
    int64_t acc = 0;
    for (long i = 0; i < s->n; i++) {
        acc += a[i];
        p[i] = acc;
    }
}

static void scan_checksum(State *s, char *out) {
    int64_t acc = 0;
    for (long i = 0; i < s->n; i++) acc += s->ib[i] % 1000003;
    sprintf(out, "%lld", (long long)acc);
}

/* ------------------------------------------------------------- hash table */

static void hash_setup(State *s) {
    long n = s->n;
    uint64_t *seq = xmalloc(sizeof(uint64_t) * 2 * n);
    uint64_t x = 42;
    for (long i = 0; i < 2 * n; i++) {
        x = x * 48271 % 2147483647;
        seq[i] = x;
    }
    s->keys = xmalloc(sizeof(uint64_t) * n);
    s->queries = xmalloc(sizeof(uint64_t) * n);
    for (long i = 0; i < n; i++) {
        s->keys[i] = seq[i];
        s->queries[i] = (i % 2 == 0) ? seq[i] : seq[n + i];
    }
    free(seq);
    s->cap = 1;
    while (s->cap < 2 * n) s->cap <<= 1;
    s->tkeys = xmalloc(sizeof(uint64_t) * s->cap);
    s->tvals = xmalloc(sizeof(int64_t) * s->cap);
}

/* Open addressing, linear probing, Fibonacci hashing. Key 0 marks empty. */
static void hash_table(State *s) {
    long n = s->n, mask = s->cap - 1;
    int shift = 64 - __builtin_ctzl((unsigned long)s->cap);
    uint64_t *tk = s->tkeys;
    int64_t *tv = s->tvals;
    memset(tk, 0, sizeof(uint64_t) * s->cap);
    for (long i = 0; i < n; i++) {
        uint64_t k = s->keys[i];
        long h = (long)((k * 0x9E3779B97F4A7C15ULL) >> shift);
        while (tk[h] != 0 && tk[h] != k) h = (h + 1) & mask;
        tk[h] = k;
        tv[h] = i;
    }
    int64_t found = 0, vsum = 0;
    for (long i = 0; i < n; i++) {
        uint64_t q = s->queries[i];
        long h = (long)((q * 0x9E3779B97F4A7C15ULL) >> shift);
        while (tk[h] != 0) {
            if (tk[h] == q) {
                found++;
                vsum += tv[h];
                break;
            }
            h = (h + 1) & mask;
        }
    }
    s->found = found;
    s->vsum = vsum;
}

/* ------------------------------------------------------------- driver */

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "noop") == 0) {
        puts("RESULT noop 0 0");
        return 0;
    }
    if (argc < 5) {
        fprintf(stderr, "usage: %s <kernel> <n> <warmup> <repeats> [threads]\n", argv[0]);
        return 1;
    }
    const char *k = argv[1];
    State s;
    memset(&s, 0, sizeof s);
    s.n = atol(argv[2]);
    int warmup = atoi(argv[3]), repeats = atoi(argv[4]);

    void (*run)(State *) = NULL;
    if (!strcmp(k, "matmul_naive")) { mat_setup(&s); run = matmul_naive; }
    else if (!strcmp(k, "matmul_tiled")) { mat_setup(&s); run = matmul_tiled; }
    else if (!strcmp(k, "mandelbrot")) { run = mandelbrot; }
    else if (!strcmp(k, "nbody")) { nbody_setup(&s); run = nbody; }
    else if (!strcmp(k, "reduction")) { int_setup(&s); run = reduction; }
    else if (!strcmp(k, "prefix_sum")) { int_setup(&s); run = prefix_sum; }
    else if (!strcmp(k, "hash_table")) { hash_setup(&s); run = hash_table; }
    else { printf("UNSUPPORTED %s\n", k); return 3; }

    for (int r = 0; r < warmup; r++) run(&s);
    double *times = malloc(sizeof(double) * (repeats > 0 ? repeats : 1));
    for (int r = 0; r < repeats; r++) {
        double t0 = now();
        run(&s);
        times[r] = now() - t0;
    }

    char cs[64];
    if (!strncmp(k, "matmul", 6)) mat_checksum(&s, cs);
    else if (!strcmp(k, "mandelbrot")) sprintf(cs, "%lld", (long long)s.total);
    else if (!strcmp(k, "nbody")) nbody_checksum(&s, cs);
    else if (!strcmp(k, "reduction")) sprintf(cs, "%lld", (long long)s.isum);
    else if (!strcmp(k, "prefix_sum")) scan_checksum(&s, cs);
    else sprintf(cs, "%lld", (long long)(s.vsum * 7 + s.found));

    printf("RESULT %s %s ", k, cs);
    for (int r = 0; r < repeats; r++) printf(r ? ",%.9f" : "%.9f", times[r]);
    printf("\n");
    return 0;
}

// main.c  (C11 + OpenMP + pthread)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <omp.h>

typedef struct {
    double mean_ms;
    double p95_ms;
    double sum;
} Stats;

static double now_ms(void) {
#if defined(CLOCK_MONOTONIC)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
#else
    // Fallback (less precise)
    return (double)clock() * 1000.0 / (double)CLOCKS_PER_SEC;
#endif
}

static int cmp_double(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da < db) ? -1 : (da > db);
}

static Stats compute_stats(double* times_ms, int n, double sum) {
    double total = 0.0;
    for (int i = 0; i < n; i++) total += times_ms[i];
    double mean = total / (double)n;

    qsort(times_ms, (size_t)n, sizeof(double), cmp_double);

    int idx = (int)ceil(0.95 * (double)n) - 1; // p95 index
    if (idx < 0) idx = 0;
    if (idx >= n) idx = n - 1;

    Stats s;
    s.mean_ms = mean;
    s.p95_ms = times_ms[idx];
    s.sum = sum;
    return s;
}

static Stats serial_compute(const double* A, size_t N) {
    double times[10];
    double sum = 0.0;

    // warmup
    for (int w = 0; w < 3; w++) {
        double temp = 0.0;
        for (size_t i = 0; i < N; i++) temp += A[i];
    }

    for (int r = 0; r < 10; r++) {
        double temp = 0.0;
        double st = now_ms();
        for (size_t i = 0; i < N; i++) temp += A[i];
        double en = now_ms();
        sum = temp;
        times[r] = en - st;
    }

    Stats s = compute_stats(times, 10, sum);
    printf("Serial: sum=%.0f mean=%.3f ms p95=%.3f ms\n", s.sum, s.mean_ms, s.p95_ms);
    return s;
}

static Stats omp_compute(const double* A, size_t N) {
    double times[10];
    double sum = 0.0;

    // warmup
    for (int w = 0; w < 3; w++) {
        double temp = 0.0;
        for (size_t i = 0; i < N; i++) temp += A[i];
    }

    for (int r = 0; r < 10; r++) {
        double temp = 0.0;
        double st = now_ms();

        #pragma omp parallel for reduction(+:temp)
        for (long long i = 0; i < (long long)N; i++) {
            temp += A[(size_t)i];
        }

        double en = now_ms();
        sum = temp;
        times[r] = en - st;
    }

    Stats s = compute_stats(times, 10, sum);
    printf("OMP   : sum=%.0f mean=%.3f ms p95=%.3f ms\n", s.sum, s.mean_ms, s.p95_ms);
    return s;
}

// ---- pthread version ----
typedef struct {
    const double* A;
    size_t start;
    size_t end;
    double* out_partial;
} WorkerArgs;

static void* thread_worker(void* arg) {
    WorkerArgs* wa = (WorkerArgs*)arg;
    double temp = 0.0;
    for (size_t i = wa->start; i < wa->end; i++) temp += wa->A[i];
    *(wa->out_partial) = temp;
    return NULL;
}

static Stats thread_compute(const double* A, size_t N) {
    double times[10];
    double sum = 0.0;

    int n_threads = omp_get_num_procs(); // cross-platform enough
    if (n_threads <= 0) n_threads = 4;

    // warmup
    for (int w = 0; w < 3; w++) {
        double temp = 0.0;
        for (size_t i = 0; i < N; i++) temp += A[i];
    }

    for (int r = 0; r < 10; r++) {
        double st = now_ms();

        pthread_t* threads = (pthread_t*)malloc((size_t)n_threads * sizeof(pthread_t));
        WorkerArgs* args   = (WorkerArgs*)malloc((size_t)n_threads * sizeof(WorkerArgs));
        double* partial    = (double*)calloc((size_t)n_threads, sizeof(double));

        size_t block = N / (size_t)n_threads;

        for (int t = 0; t < n_threads; t++) {
            size_t s = (size_t)t * block;
            size_t e = (t == n_threads - 1) ? N : (s + block);

            args[t].A = A;
            args[t].start = s;
            args[t].end = e;
            args[t].out_partial = &partial[t];

            pthread_create(&threads[t], NULL, thread_worker, &args[t]);
        }
        for (int t = 0; t < n_threads; t++) pthread_join(threads[t], NULL);

        double temp = 0.0;
        for (int t = 0; t < n_threads; t++) temp += partial[t];
        sum = temp;

        double en = now_ms();
        times[r] = en - st;

        free(threads);
        free(args);
        free(partial);
    }

    Stats s = compute_stats(times, 10, sum);
    printf("Thread: sum=%.0f mean=%.3f ms p95=%.3f ms\n", s.sum, s.mean_ms, s.p95_ms);
    return s;
}

int main(void) {
    size_t N = 80000000ULL;

    double* A = (double*)malloc(N * sizeof(double));
    if (!A) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }
    for (size_t i = 0; i < N; i++) A[i] = 1.0;

    Stats s_serial = serial_compute(A, N);
    Stats s_omp    = omp_compute(A, N);
    Stats s_thr    = thread_compute(A, N);

    double sp_omp_mean = s_serial.mean_ms / s_omp.mean_ms;
    double sp_thr_mean = s_serial.mean_ms / s_thr.mean_ms;
    double sp_omp_p95  = s_serial.p95_ms  / s_omp.p95_ms;
    double sp_thr_p95  = s_serial.p95_ms  / s_thr.p95_ms;

    printf("\nSpeedup (mean): OMP=%.3fx, Thread=%.3fx\n", sp_omp_mean, sp_thr_mean);
    printf("Speedup (p95) : OMP=%.3fx, Thread=%.3fx\n", sp_omp_p95,  sp_thr_p95);

    FILE* f = fopen("results.csv", "w");
    if (f) {
        fprintf(f, "method,mean_ms,p95_ms,speedup_mean,speedup_p95\n");
        fprintf(f, "Serial,%.6f,%.6f,1,1\n", s_serial.mean_ms, s_serial.p95_ms);
        fprintf(f, "OMP,%.6f,%.6f,%.6f,%.6f\n", s_omp.mean_ms, s_omp.p95_ms, sp_omp_mean, sp_omp_p95);
        fprintf(f, "Thread,%.6f,%.6f,%.6f,%.6f\n", s_thr.mean_ms, s_thr.p95_ms, sp_thr_mean, sp_thr_p95);
        fclose(f);
        printf("\nSaved: results.csv\n");
    } else {
        fprintf(stderr, "Could not write results.csv\n");
    }

    free(A);
    return 0;
}

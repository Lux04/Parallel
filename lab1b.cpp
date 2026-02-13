#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <cmath>
#include <omp.h>

const size_t N = 80000000;

struct Stats {
    double mean_ms = 0.0;
    double p95_ms  = 0.0;
};

static Stats stats_from_times(std::vector<double> times) {
    double total = 0.0;
    for (double t : times) total += t;
    double mean = total / (double)times.size();

    std::sort(times.begin(), times.end());
    int n = (int)times.size();
    int idx = (int)std::ceil(0.95 * n) - 1;
    if (idx < 0) idx = 0;
    if (idx >= n) idx = n - 1;

    return { mean, times[idx] };
}

// --- workload ---
static inline double work(double x) {
    return std::sin(x) * 0.5 + 0.25;
}

static void warmup(std::vector<double>& A, int w = 3) {
    for (int k = 0; k < w; k++) {
        for (size_t i = 0; i < N; i++) A[i] = work(A[i]);
    }
}

Stats serial_compute(std::vector<double> A) {
    std::vector<double> times;
    warmup(A, 3);

    for (int r = 0; r < 10; r++) {
        auto st = std::chrono::steady_clock::now();
        for (size_t i = 0; i < N; i++) A[i] = work(A[i]);
        auto en = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(en - st).count();
        times.push_back(ms);
    }
    return stats_from_times(times);
}

Stats omp_compute_n(std::vector<double> A, int n_threads) {
    if (n_threads <= 0) n_threads = 1;
    omp_set_num_threads(n_threads);

    std::vector<double> times;
    warmup(A, 3);

    for (int r = 0; r < 10; r++) {
        auto st = std::chrono::steady_clock::now();
        #pragma omp parallel for
        for (long long i = 0; i < (long long)N; i++) {
            A[(size_t)i] = work(A[(size_t)i]);
        }
        auto en = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(en - st).count();
        times.push_back(ms);
    }
    return stats_from_times(times);
}

static void thread_worker(std::vector<double>& A, size_t start, size_t end) {
    for (size_t i = start; i < end; i++) A[i] = work(A[i]);
}

Stats thread_compute_n(std::vector<double> A, int n_threads) {
    if (n_threads <= 0) n_threads = 1;

    std::vector<double> times;
    warmup(A, 3);

    size_t block = N / (size_t)n_threads;

    for (int r = 0; r < 10; r++) {
        auto st = std::chrono::steady_clock::now();

        std::vector<std::thread> threads;
        threads.reserve(n_threads);

        for (int t = 0; t < n_threads; t++) {
            size_t s = (size_t)t * block;
            size_t e = (t == n_threads - 1) ? N : (s + block);
            threads.emplace_back(thread_worker, std::ref(A), s, e);
        }
        for (auto& th : threads) th.join();

        auto en = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(en - st).count();
        times.push_back(ms);
    }

    return stats_from_times(times);
}

int main() {
    std::vector<double> base(N, 1.0);

    // Serial (baseline)
    auto s_serial = serial_compute(base);
    std::cout << "Serial mean=" << s_serial.mean_ms << " ms, p95=" << s_serial.p95_ms << " ms\n";

    int thread_list[] = {1, 2, 4, 8, 12, 16, 20, 24, 28};
    int m = sizeof(thread_list) / sizeof(thread_list[0]);

    std::ofstream out("results_sin.csv");
    out << "method,threads,mean_ms,p95_ms,speedup_mean,speedup_p95\n";
    out << "Serial,1," << s_serial.mean_ms << "," << s_serial.p95_ms << ",1,1\n";

    // OMP sweep
    for (int i = 0; i < m; i++) {
        int nt = thread_list[i];
        auto s = omp_compute_n(base, nt);

        double sp_mean = s_serial.mean_ms / s.mean_ms;
        double sp_p95  = s_serial.p95_ms  / s.p95_ms;

        std::cout << "[OMP] nt=" << nt << " mean=" << s.mean_ms << " p95=" << s.p95_ms
                  << " speedup=" << sp_mean << "x\n";

        out << "OMP," << nt << "," << s.mean_ms << "," << s.p95_ms
            << "," << sp_mean << "," << sp_p95 << "\n";
    }

    // Thread sweep
    for (int i = 0; i < m; i++) {
        int nt = thread_list[i];
        auto s = thread_compute_n(base, nt);

        double sp_mean = s_serial.mean_ms / s.mean_ms;
        double sp_p95  = s_serial.p95_ms  / s.p95_ms;

        std::cout << "[Thread] nt=" << nt << " mean=" << s.mean_ms << " p95=" << s.p95_ms
                  << " speedup=" << sp_mean << "x\n";

        out << "Thread," << nt << "," << s.mean_ms << "," << s.p95_ms
            << "," << sp_mean << "," << sp_p95 << "\n";
    }

    out.close();
    std::cout << "\nSaved: results_sin.csv\n";
    return 0;
}

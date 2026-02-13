#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <cmath>
#include <omp.h>

struct Stats {
    double mean_ms = 0.0;
    double p95_ms  = 0.0;
    double sum     = 0.0;
};

static Stats compute_stats(std::vector<double> times_ms, double sum) {
    double total = 0.0;
    for (double t : times_ms) total += t;
    double mean = total / (double)times_ms.size();

    std::sort(times_ms.begin(), times_ms.end());
    int n = (int)times_ms.size();
    int idx = (int)std::ceil(0.95 * n) - 1;
    if (idx < 0) idx = 0;
    if (idx >= n) idx = n - 1;

    Stats s;
    s.mean_ms = mean;
    s.p95_ms  = times_ms[idx];
    s.sum     = sum;
    return s;
}

Stats serial_compute(const std::vector<double>& A) {
    std::vector<double> times;
    double sum = 0.0;

    for (int w = 0; w < 3; w++) {
        double temp = 0.0;
        for (size_t i = 0; i < A.size(); i++) temp += A[i];
    }

    for (int r = 0; r < 10; r++) {
        double temp = 0.0;
        auto st = std::chrono::steady_clock::now();
        for (size_t i = 0; i < A.size(); i++) temp += A[i];
        auto en = std::chrono::steady_clock::now();
        sum = temp;

        double ms = std::chrono::duration<double, std::milli>(en - st).count();
        times.push_back(ms);
    }

    auto s = compute_stats(times, sum);
    std::cout << "Serial: sum=" << s.sum
              << " mean=" << s.mean_ms << " ms"
              << " p95="  << s.p95_ms  << " ms\n";
    return s;
}

Stats omp_compute_n(const std::vector<double>& A, int n_threads) {
    if (n_threads <= 0) n_threads = 1;

    std::vector<double> times;
    double sum = 0.0;

    omp_set_num_threads(n_threads);

    for (int w = 0; w < 3; w++) {
        double temp = 0.0;
        for (size_t i = 0; i < A.size(); i++) temp += A[i];
    }

    for (int r = 0; r < 10; r++) {
        double temp = 0.0;
        auto st = std::chrono::steady_clock::now();

        #pragma omp parallel for reduction(+:temp)
        for (long long i = 0; i < (long long)A.size(); i++) {
            temp += A[(size_t)i];
        }

        auto en = std::chrono::steady_clock::now();
        sum = temp;

        double ms = std::chrono::duration<double, std::milli>(en - st).count();
        times.push_back(ms);
    }

    return compute_stats(times, sum);
}

static void thread_worker(const std::vector<double>& A, size_t start, size_t end, double& out) {
    double temp = 0.0;
    for (size_t i = start; i < end; i++) temp += A[i];
    out = temp;
}

Stats thread_compute_n(const std::vector<double>& A, int n_threads) {
    if (n_threads <= 0) n_threads = 1;  

    std::vector<double> times;
    double sum = 0.0;

    for (int w = 0; w < 3; w++) {
        double temp = 0.0;
        for (size_t i = 0; i < A.size(); i++) temp += A[i];
    }

    for (int r = 0; r < 10; r++) {
        auto st = std::chrono::steady_clock::now();

        std::vector<std::thread> threads;
        std::vector<double> partial(n_threads, 0.0);

        size_t block = A.size() / (size_t)n_threads;

        for (int t = 0; t < n_threads; t++) {
            size_t s = (size_t)t * block;
            size_t e = (t == n_threads - 1) ? A.size() : (s + block);
            threads.emplace_back(thread_worker, std::cref(A), s, e, std::ref(partial[t]));
        }
        for (auto& th : threads) th.join();

        double temp = 0.0;
        for (double x : partial) temp += x;
        sum = temp;

        auto en = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(en - st).count();
        times.push_back(ms);
    }

    return compute_stats(times, sum);
}

int main() {
    const size_t N = 80000000;
    std::vector<double> A(N, 1.0);

    auto s_serial = serial_compute(A);

    std::ofstream out("results_all.csv");
    out << "method,threads,mean_ms,p95_ms,speedup_mean,speedup_p95\n";
    out << "Serial,1," << s_serial.mean_ms << "," << s_serial.p95_ms << ",1,1\n";

    int thread_list[] = {1, 2, 4, 8, 12, 16, 20, 24, 28};
    int m = sizeof(thread_list) / sizeof(thread_list[0]);

    for (int i = 0; i < m; i++) {
        int nt = thread_list[i];
        auto s_omp = omp_compute_n(A, nt);

        double sp_mean = s_serial.mean_ms / s_omp.mean_ms;
        double sp_p95  = s_serial.p95_ms  / s_omp.p95_ms;

        std::cout << "[OMP] threads=" << nt
                  << " mean=" << s_omp.mean_ms << " ms"
                  << " p95="  << s_omp.p95_ms  << " ms"
                  << " speedup=" << sp_mean << "x\n";

        out << "OMP," << nt << "," << s_omp.mean_ms << "," << s_omp.p95_ms
            << "," << sp_mean << "," << sp_p95 << "\n";
    }

    for (int i = 0; i < m; i++) {
        int nt = thread_list[i];
        auto s_thr = thread_compute_n(A, nt);

        double sp_mean = s_serial.mean_ms / s_thr.mean_ms;
        double sp_p95  = s_serial.p95_ms  / s_thr.p95_ms;

        std::cout << "[Thread] threads=" << nt
                  << " mean=" << s_thr.mean_ms << " ms"
                  << " p95="  << s_thr.p95_ms  << " ms"
                  << " speedup=" << sp_mean << "x\n";

        out << "Thread," << nt << "," << s_thr.mean_ms << "," << s_thr.p95_ms
            << "," << sp_mean << "," << sp_p95 << "\n";
    }

    out.close();
    std::cout << "\nSaved: results_all.csv\n";
    return 0;
}

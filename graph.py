import csv
import matplotlib.pyplot as plt

omp_t, omp_s = [], []
thr_t, thr_s = [], []

with open("results_all.csv") as f:
    r = csv.DictReader(f)
    for row in r:
        if row["method"] == "OMP":
            omp_t.append(int(row["threads"]))
            omp_s.append(float(row["speedup_mean"]))
        elif row["method"] == "Thread":
            thr_t.append(int(row["threads"]))
            thr_s.append(float(row["speedup_mean"]))

# sort
omp = sorted(zip(omp_t, omp_s))
thr = sorted(zip(thr_t, thr_s))

omp_t, omp_s = zip(*omp)
thr_t, thr_s = zip(*thr)

plt.plot(omp_t, omp_s, marker="o", label="OpenMP")
plt.plot(thr_t, thr_s, marker="o", label="std::thread")
plt.xlabel("Threads")
plt.ylabel("Speedup (x)")
plt.title("Scalability: OMP vs std::thread")
plt.grid(True)
plt.legend()
plt.savefig("scalability.png")
plt.show()

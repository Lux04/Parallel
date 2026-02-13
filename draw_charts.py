import pandas as pd
import matplotlib.pyplot as plt

def plot_scalability(csv_file, title, output_name):
    try:
        df = pd.read_csv(csv_file)
        
        # 1-ээс бага thread-тэй мөр байвал устгах (алдаанаас сэргийлнэ)
        df = df[df['threads'] >= 1]

        plt.figure(figsize=(12, 5))

        # 1. Speedup График
        plt.subplot(1, 2, 1)
        for method in df['method'].unique():
            subset = df[df['method'] == method]
            plt.plot(subset['threads'], subset['speedup_mean'], marker='o', label=f'{method} Speedup')
        
        plt.plot(subset['threads'], subset['threads'], '--', color='gray', label='Ideal (Linear)')
        plt.title(f'Speedup: {title}')
        plt.xlabel('Number of Threads')
        plt.ylabel('Speedup (x)')
        plt.legend()
        plt.grid(True)

        # 2. Efficiency График (E = Speedup / Threads)
        plt.subplot(1, 2, 2)
        for method in df['method'].unique():
            subset = df[df['method'] == method]
            efficiency = subset['speedup_mean'] / subset['threads']
            plt.plot(subset['threads'], efficiency, marker='s', label=f'{method} Efficiency')
        
        plt.axhline(y=1.0, color='gray', linestyle='--', label='Ideal (100%)')
        plt.title(f'Efficiency: {title}')
        plt.xlabel('Number of Threads')
        plt.ylabel('Efficiency (0.0 - 1.0)')
        plt.ylim(0, 1.1)
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(output_name)
        plt.show()
        print(f"Амжилттай хадгалагдлаа: {output_name}")

    except Exception as e:
        print(f"Алдаа гарлаа: {e}")

# Ашиглах хэсэг:
plot_scalability('results_all.csv', 'Summation (Task A)', 'final_sum_report.png')
plot_scalability('results_sin.csv', 'Sin Transformation (Task B)', 'final_sin_report.png')
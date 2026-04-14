%%writefile lab3.cu

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda_runtime.h>

// --- 1-Р ХЭСЭГ: SAXPY КЕРНЕЛ ---
__global__ void saxpy_kernel(int N, float A, float *x, float *y, float *r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) r[i] = A * x[i] + y[i];
}

// --- 2-Р ХЭСЭГ: БЭЛТГЭЛ КЕРНЕЛҮҮД ---
__global__ void exclusive_scan_kernel(int *d_data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Up-sweep
    for (int d = 1; d < n; d <<= 1) {
        int index = (i + 1) * 2 * d - 1;
        if (index < n) d_data[index] += d_data[index - d];
        __syncthreads();
    }
    // Down-sweep
    if (i == 0) d_data[n - 1] = 0;
    __syncthreads();
    for (int d = n >> 1; d > 0; d >>= 1) {
        int index = (i + 1) * 2 * d - 1;
        if (index < n) {
            int t = d_data[index - d];
            d_data[index - d] = d_data[index];
            d_data[index] += t;
        }
        __syncthreads();
    }
}

__global__ void find_repeats_kernel(int *A, int *F, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N - 1) F[i] = (A[i] == A[i+1]) ? 1 : 0;
    else if (i == N - 1) F[i] = 0;
}

__global__ void scatter_kernel(int *A, int *F, int *S, int *Output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && F[i] == 1) Output[S[i]] = i; // Индексийг буцаана
}

int main() {
    // --- 1-Р ХЭСЭГ: SAXPY & BANDWIDTH ---
    int N = 1000000;
    size_t size = N * sizeof(float);
    float *h_x = (float*)malloc(size); float *h_y = (float*)malloc(size); float *h_r = (float*)malloc(size);
    for(int i=0; i<N; i++) { h_x[i]=1.0f; h_y[i]=2.0f; }
    
    float *d_x, *d_y, *d_r;
    cudaMalloc(&d_x, size); cudaMalloc(&d_y, size); cudaMalloc(&d_r, size);
    
    // Нийт хугацаа эхлэх
    auto total_start = std::chrono::steady_clock::now();
    
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);
    
    auto kernel_start = std::chrono::steady_clock::now();
    saxpy_kernel<<< (N+255)/256, 256 >>>(N, 2.0f, d_x, d_y, d_r);
    cudaDeviceSynchronize(); // Kernel хэмжилтийн хувьд зайлшгүй
    auto kernel_end = std::chrono::steady_clock::now();
    
    cudaMemcpy(h_r, d_r, size, cudaMemcpyDeviceToHost);
    auto total_end = std::chrono::steady_clock::now();
    
    double k_time = std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count();
    printf("SAXPY Kernel time: %f ms\n", k_time);
    printf("Bandwidth: %f GB/s\n", (3.0 * size) / (k_time / 1000.0) / 1e9);

    // --- 2-Р ХЭСЭГ: FIND REPEATS ---
    int arr[] = {1,2,2,1,1,1,3,5,3,3};
    int n2 = 10;
    int *d_A, *d_F, *d_S, *d_Out;
    cudaMalloc(&d_A, n2*sizeof(int)); cudaMalloc(&d_F, n2*sizeof(int));
    cudaMalloc(&d_S, n2*sizeof(int)); cudaMalloc(&d_Out, n2*sizeof(int));
    cudaMemset(d_Out, 0, n2*sizeof(int)); // 0-ээр дүүргэх
    
    cudaMemcpy(d_A, arr, n2*sizeof(int), cudaMemcpyHostToDevice);
    find_repeats_kernel<<<1, n2>>>(d_A, d_F, n2);
    cudaMemcpy(d_S, d_F, n2*sizeof(int), cudaMemcpyDeviceToDevice);
    exclusive_scan_kernel<<<1, n2>>>(d_S, n2);
    scatter_kernel<<<1, n2>>>(d_A, d_F, d_S, d_Out, n2);
    
    int *h_out = (int*)malloc(n2*sizeof(int));
    cudaMemcpy(h_out, d_Out, n2*sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Find Repeats index: ");
    for(int i=0; i<4; i++) printf("%d ", h_out[i]); 
    printf("\n");

    // Чөлөөлөх
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_r);
    cudaFree(d_A); cudaFree(d_F); cudaFree(d_S); cudaFree(d_Out);
    free(h_x); free(h_y); free(h_r); free(h_out);
    return 0;
}
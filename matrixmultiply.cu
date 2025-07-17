#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define N 256
void cpu_matrixmul(int n, float* a, float* b, float* c) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += a[i * n + k] * b[k * n + j];
                // sum += A[i][k] * B[k][j]
            }
            c[i * n + j] = sum;
        }
    }
}

int main() {
    float* A = (float*)malloc(N * N * sizeof(float));
    float* B = (float*)malloc(N * N * sizeof(float));
    float* C_cpu = (float*)malloc(N * N * sizeof(float));

    for (int i = 0; i < N*N; i++) { //initializing
        A[i] = 2.0f;
        B[i] = 3.0f;
    }

    auto start = std::chrono::high_resolution_clock::now();
    cpu_matrixmul(N, A, B, C_cpu);
    auto end = std::chrono::high_resolution_clock::now();
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    std::cout << "CPU execution time: " << microseconds << " microseconds" << std::endl;

    free(A);
    free(B);
    free(C_cpu);
}
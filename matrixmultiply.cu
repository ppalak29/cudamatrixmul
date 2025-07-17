#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define N 1024
#define BLOCK_SIZE 16

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

__global__ 
void cuda_matrixmul(int n, float* a, float*b, float* c) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; //x goes across columns
    int row = blockIdx.y * blockDim.y + threadIdx.y; //y goes across rows
    //row and column correspond to c[row][col] so does entire computation for this

    if (row < n && col < n) { //this runs sequentially
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += a[row * n + k] * b[k * n + col];
                // sum += A[i][k] * B[k][j]
        }
        c[row * n + col] = sum;
    }
}

int main() {
    int size = N * N * sizeof(float);
    float* A, float* B, float* C_cpu, float* C_gpu;
    float* d_A, float* d_B, float* d_C;
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C_cpu = (float*)malloc(size);
    C_gpu = (float*)malloc(size);

    for (int i = 0; i < N*N; i++) { //initializing
        A[i] = 2.0f;
        B[i] = 3.0f;
    }

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();
    cpu_matrixmul(N, A, B, C_cpu);
    auto end = std::chrono::high_resolution_clock::now();
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    std::cout << "CPU execution time: " << microseconds << " microseconds" << std::endl;

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1)/BLOCK_SIZE, (N + BLOCK_SIZE - 1)/BLOCK_SIZE);

    start = std::chrono::high_resolution_clock::now();
    //how do i know the threads here are all complete
    cuda_matrixmul<<<gridSize, blockSize>>>(N, d_A, d_B, d_C);
    end = std::chrono::high_resolution_clock::now();
    microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    std::cout << "GPU execution time: " << microseconds << " microseconds" << std::endl;
    cudaMemcpy(C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C_cpu);
    free(C_gpu);
}
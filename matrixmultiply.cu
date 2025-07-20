#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define N 1024
#define BLOCK_SIZE 16 //most popular, esp for tiling

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

__global__
void cuda_tiled_matrixmul(int n, float* a, float* b, float* c) {
    __shared__ float tile_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tile_b[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty; //global row for c
    int col = blockIdx.x * BLOCK_SIZE + tx; //global col for c

    float sum = 0.0f;

    for (int tile = 0; tile < ((n + BLOCK_SIZE - 1) / BLOCK_SIZE); tile++) {
        //load tile_a
        int a_col = tile * BLOCK_SIZE + tx;
        if (row < n && a_col < n) {
            tile_a[ty][tx] = a[row * n + a_col];
        }
        else {
            tile_a[ty][tx] = 0.0f;
        }

        //load tile_b
        int b_row = tile * BLOCK_SIZE + ty;
        if (col < n && b_row < n) {
            tile_b[ty][tx] = b[b_row * n + col];
        }
        else {
            tile_a[ty][tx] = 0.0f;
        }

        __syncthreads(); // wait for tile_a and tile_b to be fully loaded

        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += tile_a[ty][k] * tile_b[k][tx];
        }
        __syncthreads(); // wait for sum to be calculated before moving tiles
    }

    if (row < n && col < n) { // in the case where more threads than elements
        c[row * n + col] = sum;
    }

}

int main() {
    int size = N * N * sizeof(float);
    float* A, float* B, float* C_cpu, float* C_gpu, float* C_gpu_tiled;
    float* d_A, float* d_B, float* d_C;
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C_cpu = (float*)malloc(size);
    C_gpu = (float*)malloc(size);
    C_gpu_tiled = (float*)malloc(size);

    for (int i = 0; i < N*N; i++) { //initializing
        A[i] = 2.0f;
        B[i] = 3.0f;
    }

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // cpu baseline
    auto start = std::chrono::high_resolution_clock::now();
    cpu_matrixmul(N, A, B, C_cpu);
    auto end = std::chrono::high_resolution_clock::now();
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    std::cout << "CPU execution time for matrix mul: " << microseconds << " microseconds" << std::endl;

    // basic cuda
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1)/BLOCK_SIZE, (N + BLOCK_SIZE - 1)/BLOCK_SIZE);

    start = std::chrono::high_resolution_clock::now();
    cuda_matrixmul<<<gridSize, blockSize>>>(N, d_A, d_B, d_C);
    end = std::chrono::high_resolution_clock::now();
    microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    std::cout << "GPU execution time for matrix mul: " << microseconds << " microseconds" << std::endl;
    cudaMemcpy(C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    // cuda tiled
    dim3 tiledblockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 tiledgridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    start = std::chrono::high_resolution_clock::now();
    cuda_tiled_matrixmul<<<tiledblockSize, tiledgridSize>>>(N, d_A, d_B, d_C);
    end = std::chrono::high_resolution_clock::now();
    microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    std::cout << "GPU execution time for tiled matrix mul: " << microseconds << " microseconds" << std::endl;
    cudaMemcpy(C_gpu_tiled, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << C_cpu[0] << std::endl;
    std::cout << C_gpu[0] << std::endl;
    std::cout << C_gpu_tiled[0] << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C_cpu);
    free(C_gpu);
    free(C_gpu_tiled);
}
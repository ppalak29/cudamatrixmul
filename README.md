# CUDA Matrix Multiplication Performance Analysis

## Project Overview
Implements and compares three matrix multiplication approaches on 1024×1024 matrices using Nsight Compute for performance analysis:

CPU Baseline: Traditional O(N³) implementation
Naive CUDA: Direct GPU parallelization
Tiled CUDA: Memory-optimized GPU implementation using shared memory

## Performance Results

| **Implementation** | **Time (s)** | **Speedup** | **GFLOPS** |
|--------------------|--------------|-------------|------------|
| CPU Baseline       | 23.67        | 1.0×        | 0.09       |
| Naive CUDA         | 0.153        | 155×        | 14.1       |
| Tiled CUDA         | 0.0004       | 53,118×     | 4,847      |

Matrix size: 1024×1024 (2.15 billion operations)

## Implementation Analysis
### CPU Implementation
#### Advantages:
1. Simple, portable, no GPU required
2. Predictable performance and debugging
####Tradeoffs:
1. Single-threaded, severely limited scalability
1. Underutilizes modern parallel hardware

### Naive CUDA
#### Advantages:
1. Massive parallelization (65,536 threads)
2. Significant speedup with minimal code changes
#### Tradeoffs:
1. Poor memory access patterns (strided B matrix access)
2. Low memory bandwidth utilization (~25% of peak)
3. Memory-bound performance bottleneck

### Tiled CUDA
#### Advantages:
1. Shared memory optimization (100x faster than global memory)
2. Coalesced memory access patterns
3. High GPU utilization (>85% compute and memory)
4. Each data element reused 16x, minimizing global memory traffic
#### Tradeoffs:
1. Complex implementation with synchronization requirements
2. Limited by shared memory size (48KB per block)
3. Requires careful tile size tuning
4. Build & Run

## Compile
nvcc -O3 -arch=sm_75 matrix_multiply.cu -o matrix_multiply

## Run
./matrix_multiply

## Profile with Nsight
nsys profile --stats=true ./matrix_multiply
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ./matrix_multiply
Requirements: CUDA Toolkit 11.0+, GPU with Compute Capability 3.0+


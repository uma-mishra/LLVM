#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <cuda.h>

// Kernel to initialize arrays
extern "C" __global__ void initKernel(unsigned V, bool *init_array, bool initVal) {
    unsigned id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id < V) {
        init_array[id] = initVal;
    }
}

// SSSP Kernel
extern "C" __global__ void Compute_SSSP_kernel(int *gpu_OA, int *gpu_edgeList, int *gpu_weight, int *gpu_dist, int V, int MAX_VAL,
                                    bool *gpu_modified_prev, bool *gpu_modified_next, bool *gpu_finished) {
    unsigned int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id < V && gpu_modified_prev[id]) {
        for (int edge = gpu_OA[id]; edge < gpu_OA[id + 1]; edge++) {
            int nbr = gpu_edgeList[edge];
            int dist_new = gpu_dist[id] + gpu_weight[edge];

            if (gpu_dist[nbr] > dist_new) {
                atomicMin(&gpu_dist[nbr], dist_new);
                gpu_modified_next[nbr] = true;
                gpu_finished[0] = false;
            }
        }
    }
}


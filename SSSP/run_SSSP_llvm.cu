#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA_ERROR(call) { \
    CUresult err = call; \
    if (err != CUDA_SUCCESS) { \
        const char *errStr; \
        cuGetErrorString(err, &errStr); \
        fprintf(stderr, "CUDA Error: %s\n", errStr); \
        exit(EXIT_FAILURE); \
    } \
}

int main() {
    // Initialize the CUDA driver
    CHECK_CUDA_ERROR(cuInit(0));

    // Get the first CUDA device
    CUdevice device;
    CHECK_CUDA_ERROR(cuDeviceGet(&device, 0));

    // Create a CUDA context
    CUcontext context;
    CHECK_CUDA_ERROR(cuCtxCreate(&context, 0, device));

    // Load the PTX file
    CUmodule module;
    CHECK_CUDA_ERROR(cuModuleLoad(&module, "SSSP.ptx"));

    // Get the kernel functions
    CUfunction init_func;
    CUfunction sssp_func;
    CHECK_CUDA_ERROR(cuModuleGetFunction(&init_func, module, "initKernel"));
    CHECK_CUDA_ERROR(cuModuleGetFunction(&sssp_func, module,"Compute_SSSP_kernel"));

    // Sample graph data (replace with actual graph input if needed)
    int V = 5;
    int E = 8;
    int OA[] = {0, 2, 4, 5, 7, 8};
    int edgeList[] = {1, 2, 0, 3, 2, 1, 4, 3};
    int weight[] = {2, 4, 1, 7, 3, 2, 1, 5};
    int src = 0;

    int MAX_VAL = INT_MAX;
    int *dist = (int*)malloc(V * sizeof(int));
    bool *modified_prev = (bool*)malloc(V * sizeof(bool));
    bool *modified_next = (bool*)malloc(V * sizeof(bool));
    bool finished = false;

    // Initialize arrays
    for (int i = 0; i < V; ++i) {
        dist[i] = MAX_VAL;
        modified_prev[i] = false;
        modified_next[i] = false;
    }
    dist[src] = 0;
    modified_prev[src] = true;

    // Allocate device memory
    CUdeviceptr gpu_OA, gpu_edgeList, gpu_weight, gpu_dist, gpu_modified_prev, gpu_modified_next, gpu_finished;
    CHECK_CUDA_ERROR(cuMemAlloc(&gpu_OA, sizeof(int) * (V + 1)));
    CHECK_CUDA_ERROR(cuMemAlloc(&gpu_edgeList, sizeof(int) * E));
    CHECK_CUDA_ERROR(cuMemAlloc(&gpu_weight, sizeof(int) * E));
    CHECK_CUDA_ERROR(cuMemAlloc(&gpu_dist, sizeof(int) * V));
    CHECK_CUDA_ERROR(cuMemAlloc(&gpu_modified_prev, sizeof(bool) * V));
    CHECK_CUDA_ERROR(cuMemAlloc(&gpu_modified_next, sizeof(bool) * V));
    CHECK_CUDA_ERROR(cuMemAlloc(&gpu_finished, sizeof(bool)));

    // Copy data to device
    CHECK_CUDA_ERROR(cuMemcpyHtoD(gpu_OA, OA, sizeof(int) * (V + 1)));
    CHECK_CUDA_ERROR(cuMemcpyHtoD(gpu_edgeList, edgeList, sizeof(int) * E));
    CHECK_CUDA_ERROR(cuMemcpyHtoD(gpu_weight, weight, sizeof(int) * E));
    CHECK_CUDA_ERROR(cuMemcpyHtoD(gpu_dist, dist, sizeof(int) * V));
    CHECK_CUDA_ERROR(cuMemcpyHtoD(gpu_modified_prev, modified_prev, sizeof(bool) * V));

    // Set kernel launch parameters
    int block_size = 1024;
    int num_blocks = (V + block_size - 1) / block_size;

    // Run the SSSP algorithm
    do {
        finished = true;
        CHECK_CUDA_ERROR(cuMemcpyHtoD(gpu_finished, &finished, sizeof(bool)));

       // void *args1[] = {&gpu_finished, &V, &gpu_modified_prev, &modified_prev[0]};
        void *args1[] = { &V, &gpu_modified_prev, &modified_prev[0]};
        void *args2[] = {&gpu_OA, &gpu_edgeList, &gpu_weight, &gpu_dist, &V, &MAX_VAL, &gpu_modified_prev, &gpu_modified_next, &gpu_finished};

        CHECK_CUDA_ERROR(cuLaunchKernel(init_func, 1, 1, 1, 1, 1, 1, 0, 0, args1, 0));
        CHECK_CUDA_ERROR(cuLaunchKernel(sssp_func, num_blocks, 1, 1, block_size, 1, 1, 0, 0, args2, 0));

        CHECK_CUDA_ERROR(cuMemcpyDtoH(&finished, gpu_finished, sizeof(bool)));

        // Swap pointers
        CUdeviceptr temp = gpu_modified_prev;
        gpu_modified_prev = gpu_modified_next;
        gpu_modified_next = temp;
    } while (!finished);

    // Copy results back to host
    CHECK_CUDA_ERROR(cuMemcpyDtoH(dist, gpu_dist, sizeof(int) * V));

    // Print results
    for (int i = 0; i < V; ++i) {
        printf("Node %d: Distance = %d\n", i, dist[i]);
    }

    // Clean up
    cuMemFree(gpu_OA);
    cuMemFree(gpu_edgeList);
    cuMemFree(gpu_weight);
    cuMemFree(gpu_dist);
    cuMemFree(gpu_modified_prev);
    cuMemFree(gpu_modified_next);
    cuMemFree(gpu_finished);
    cuModuleUnload(module);
    cuCtxDestroy(context);

    free(dist);
    free(modified_prev);
    free(modified_next);

    return 0;
}


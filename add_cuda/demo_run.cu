#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA_ERROR(call) {                                \
    CUresult err = call;                                        \
    if (err != CUDA_SUCCESS) {                                 \
        const char *errStr;                                    \
        cuGetErrorString(err, &errStr);                       \
        fprintf(stderr, "CUDA Error: %s\n", errStr);          \
        exit(EXIT_FAILURE);                                   \
    }                                                          \
}

int main() {
    // Initialize the CUDA driver API
    CHECK_CUDA_ERROR(cuInit(0));

    // Get the first CUDA device
    CUdevice device;
    CHECK_CUDA_ERROR(cuDeviceGet(&device, 0));

    // Create a CUDA context
    CUcontext context;
    CHECK_CUDA_ERROR(cuCtxCreate(&context, 0, device));

    // Load the PTX file
    CUmodule module;
    CHECK_CUDA_ERROR(cuModuleLoad(&module, "demo.ptx"));

    // Get the kernel function
    CUfunction add_func;
    CHECK_CUDA_ERROR(cuModuleGetFunction(&add_func, module, "add"));

    // Allocate device memory
    int c = 0;
    CUdeviceptr d_c;
    CHECK_CUDA_ERROR(cuMemAlloc(&d_c, sizeof(int)));

    // Set kernel parameters
    int a = 1, b = 2;
    void *args[] = { &d_c, &a, &b };

    // Launch the kernel
    CHECK_CUDA_ERROR(cuLaunchKernel(
        add_func,
        1, 1, 1,     // Grid dimensions
        1, 1, 1,     // Block dimensions
        0, 0,        // Shared memory size and stream
        args, 0      // Kernel arguments and extra options
    ));

    // Copy the result back to the host
    CHECK_CUDA_ERROR(cuMemcpyDtoH(&c, d_c, sizeof(int)));

    // Print the result
    printf("Result: %d\n", c);

    // Clean up
    cuMemFree(d_c);
    cuModuleUnload(module);
    cuCtxDestroy(context);

    return 0;
}


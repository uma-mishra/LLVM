#include <cuda.h>
#include <iostream>
#include <vector>

#define N 1024
#define CHECK(call)                                                         \
    do {                                                                    \
        CUresult err = (call);                                              \
        if (err != CUDA_SUCCESS) {                                          \
            const char *errStr;                                             \
            cuGetErrorString(err, &errStr);                                 \
            std::cerr << "CUDA Error: " << errStr << " at line " << __LINE__ << "\n"; \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main() {
    // Initialize CUDA Driver API
    CHECK(cuInit(0));

    CUdevice device;
    CHECK(cuDeviceGet(&device, 0));

    CUcontext context;
    CHECK(cuCtxCreate(&context, 0, device));

    // Load PTX
    CUmodule module;
    CHECK(cuModuleLoad(&module, "vector_add.ptx"));

    CUfunction kernel;
    CHECK(cuModuleGetFunction(&kernel, module, "vector_add"));

    // Host memory
    std::vector<float> h_A(N), h_B(N), h_C(N);
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = N - i;
    }

    // Device memory
    CUdeviceptr d_A, d_B, d_C;
    size_t size = N * sizeof(float);

    CHECK(cuMemAlloc(&d_A, size));
    CHECK(cuMemAlloc(&d_B, size));
    CHECK(cuMemAlloc(&d_C, size));

    CHECK(cuMemcpyHtoD(d_A, h_A.data(), size));
    CHECK(cuMemcpyHtoD(d_B, h_B.data(), size));

    // Set kernel args
    void* args[] = { &d_A, &d_B, &d_C, (void*)N };

    // Launch kernel (N threads in total, 256 per block)
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    CHECK(cuLaunchKernel(kernel,
                         blocks, 1, 1,
                         threadsPerBlock, 1, 1,
                         0, 0, args, 0));

    // Copy result back to host
    CHECK(cuMemcpyDtoH(h_C.data(), d_C, size));

    // Print first 10 results
    for (int i = 0; i < 10; ++i) {
        std::cout << "C[" << i << "] = " << h_C[i] << std::endl;
    }

    // Cleanup
    cuMemFree(d_A);
    cuMemFree(d_B);
    cuMemFree(d_C);
    cuModuleUnload(module);
    cuCtxDestroy(context);

    return 0;
}

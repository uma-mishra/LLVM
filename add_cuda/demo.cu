// simple_add.cu
extern "C" __global__ void add(int *c, int a, int b) {
        *c = a + b;
}


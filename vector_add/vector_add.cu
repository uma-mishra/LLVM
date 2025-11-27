extern "C"

__global__ void vector_add(float *A , float * B, float *C, int N)

{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < N)
	{
		C[id] = A[id] + B[id];
	}

}

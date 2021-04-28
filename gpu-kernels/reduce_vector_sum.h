#include "utils.h"

__device__ __inline__ static void warp_reduce(volatile float* s_data, int tid) {
	s_data[tid] += s_data[tid + 32];
	s_data[tid] += s_data[tid + 16];
	s_data[tid] += s_data[tid + 8];
	s_data[tid] += s_data[tid + 4];
	s_data[tid] += s_data[tid + 2];
	s_data[tid] += s_data[tid + 1];
}

__global__ static void reduce_vector_sum_kernel(float* d_data_ext, int N, int dims) {
	int tid = threadIdx.x;
	int idx = blockIdx.x*(blockDim.x * 2) + threadIdx.x;
	int d = blockIdx.y*blockDim.y + threadIdx.y;
	extern __shared__ float s_data[]; // [blockDim.x]

	// init shared memory
	s_data[tid] = 0;

	// load data to shared memory
	if (idx < N) {
		s_data[tid] = d_data_ext[idx*dims + d];
		if (idx + blockDim.x < N)
			s_data[tid] += d_data_ext[(idx + blockDim.x)*dims + d];
	}

	// reduce sum
	for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
		__syncthreads();
		if (tid < stride)
			s_data[tid] += s_data[tid + stride];
	}
	__syncthreads();
	if (tid < 32)
		warp_reduce(s_data, tid);

	// copy back
	if (tid == 0) {
		d_data_ext[(N + blockIdx.x)*dims + d] = s_data[0];
	}
}


template<int block_size>
static int reduce_vector_sum(float* d_data_ext, float* h_o_data, int N, int dims) {
	int N_remain = N;
	float* d_data_ext_tmp = d_data_ext;
	while (N_remain > 1) {
		reduce_vector_sum_kernel
			<< < dim3(DIV_CEIL(N_remain, block_size * 2), dims), block_size, block_size * sizeof(float) >> >
			(d_data_ext_tmp, N_remain, dims);
		gpuErrchk;
		d_data_ext_tmp += N_remain * dims;
		N_remain = DIV_CEIL(N_remain, block_size * 2);
	}
	cudaMemcpy(h_o_data, d_data_ext_tmp, dims * sizeof(float), cudaMemcpyDeviceToHost);
	gpuErrchk;
	return cudaSuccess;
}

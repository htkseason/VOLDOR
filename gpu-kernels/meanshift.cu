#include "utils.h"
#include "gpu_kernels.h"
#include "reduce_vector_sum.h"

#define MAX_DIMS 16

#define N_THREADS 256

__constant__ static float c_mean[MAX_DIMS];


template <bool compute_weight_only = false>
__global__ static void compute_weighted_space(
	float* d_space, float kernel_var, float* d_o_weight, float* d_o_weighted_space, int N, int dims) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < N) {
		float l2_distance_sqr = 0;
		float weight = 0;

		for (int d = 0; d < dims; d++)
			l2_distance_sqr += SQR(d_space[idx*dims + d] - c_mean[d]);

		weight = expf(-l2_distance_sqr / (2 * kernel_var));
		d_o_weight[idx] = weight;
		if (!compute_weight_only) {
			for (int d = 0; d < dims; d++)
				d_o_weighted_space[idx*dims + d] = d_space[idx*dims + d] * weight;
		}
	}
}


int meanshift_gpu(float* h_space, float kernel_var,
	float* h_io_mean, float* h_o_confidence, int* used_iters,
	bool use_external_init_mean, int N, int dims,
	float epsilon, int max_iters,
	int max_init_trials, float good_init_confidence) {

	float* d_space;
	float* d_weight;
	float* d_weighted_space;
	float* ht_mean; // host temp memory for mean. size=dims
	float* ht_weight; // host temp memory for weight. size=1
#if 0
	// pinned memory seems slower if only have small number of iterations
	cudaMallocHost(&ht_mean, (dims + 1) * sizeof(float));
	ht_weight = &ht_mean[dims];
	gpuErrchk;
#else
	ht_mean = new float[dims];
	ht_weight = new float[1];
#endif

	// copy space data
	cudaMalloc((void**)&d_space, N * dims * sizeof(float));
	cudaMemcpy(d_space, h_space, N * dims * sizeof(float), cudaMemcpyHostToDevice);
	gpuErrchk;

	// allocate device buffer (weight map & weighted space map)
	int N_reduce_sum_ext = 0;
	for (int N_remain = N; N_remain > 1; N_remain = DIV_CEIL(N_remain, 2 * N_THREADS))
		N_reduce_sum_ext += DIV_CEIL(N_remain, 2 * N_THREADS);
	cudaMalloc((void**)&d_weight, (N + N_reduce_sum_ext) * sizeof(float));
	cudaMalloc((void**)&d_weighted_space, (N + N_reduce_sum_ext) * dims * sizeof(float));
	gpuErrchk;

	// init mean with external given
	if (use_external_init_mean)
		cudaMemcpyToSymbol(c_mean, h_io_mean, dims * sizeof(float));
	else { // init mean with trial of maximum confidence

		float best_init_confidence = 0;
		int best_init_sample_idx = -1;
		for (int trial = 0; trial < max_init_trials; trial++) {
			int idx_rand = (rand() % N);

			cudaMemcpyToSymbol(c_mean, &h_space[idx_rand*dims], dims * sizeof(float));
			gpuErrchk;

			// weight each sample
			compute_weighted_space<true> << <DIV_CEIL(N, N_THREADS), N_THREADS >> > (d_space, kernel_var, d_weight, NULL, N, dims);
			// sum up weight
			reduce_vector_sum<N_THREADS>(d_weight, ht_weight, N, 1);
			gpuErrchk;

			if (*ht_weight > best_init_confidence) {
				best_init_confidence = *ht_weight;
				best_init_sample_idx = idx_rand;
			}

			if (best_init_confidence > good_init_confidence*N)
				break;
		}
		cudaMemcpyToSymbol(c_mean, &h_space[best_init_sample_idx*dims], dims * sizeof(float));
		gpuErrchk;
	}

	if (used_iters != NULL)
		*used_iters = 0;

	// start iteration
	for (int iter = 0; iter < max_iters; iter++) {

		compute_weighted_space << <DIV_CEIL(N, N_THREADS), N_THREADS >> > (d_space, kernel_var, d_weight, d_weighted_space, N, dims);
		reduce_vector_sum<N_THREADS>(d_weight, ht_weight, N, 1);
		reduce_vector_sum<N_THREADS>(d_weighted_space, ht_mean, N, dims);
		gpuErrchk;

		for (int d = 0; d < dims; d++)
			ht_mean[d] = ht_mean[d] / *ht_weight;

		// update confidence & used_iter
		if (h_o_confidence != NULL)
			*h_o_confidence = *ht_weight / N;
		if (used_iters != NULL)
			*used_iters = iter + 1;

		// compute displacement
		float displacement = 0;
		for (int d = 0; d < dims; d++)
			displacement += SQR(h_io_mean[d] - ht_mean[d]);
		displacement = sqrt(displacement);

		// after compute displacement, assign back mean
		for (int d = 0; d < dims; d++)
			h_io_mean[d] = ht_mean[d];

		if (displacement < epsilon)
			break;

		cudaMemcpyToSymbol(c_mean, h_io_mean, dims * sizeof(float));
		gpuErrchk;
	}

#if 0
	cudaFreeHost(ht_mean);
	cudaFreeHost(ht_weight);
#else
	delete[] ht_mean;
	delete[] ht_weight;
#endif

	cudaFree(d_space);
	cudaFree(d_weight);
	cudaFree(d_weighted_space);
	gpuErrchk;

	return cudaSuccess;
}

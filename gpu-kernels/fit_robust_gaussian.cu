#include "utils.h"
#include "gpu_kernels.h"
#include "reduce_vector_sum.h"
#include "aux_funs.h"

#define MAX_DIMS 6
#define N_THREADS 256
//#define FULL_LEDOIT_WOLF_ESTIMATOR

__constant__ static float c_mean[MAX_DIMS];
__constant__ static float c_covar_inv[(MAX_DIMS*MAX_DIMS + MAX_DIMS) / 2];
#ifdef FULL_LEDOIT_WOLF_ESTIMATOR
__constant__ static float c_covar[(MAX_DIMS*MAX_DIMS + MAX_DIMS) / 2];
#endif
//__constant__ static float c_A; // gaussian normalize constant

template <typename T1, typename T2>
static void covar_half_to_full(const T1* half, T2* full, const int dims) {
	for (int d1 = 0; d1 < dims; d1++) {
		for (int d2 = 0; d2 <= d1; d2++) {
			full[d1*dims + d2] = (T2)half[(d1*d1 + d1) / 2 + d2];
			if (d1 != d2)
				full[d2*dims + d1] = full[d1*dims + d2];
		}
	}
}
template <typename T1, typename T2>
static void covar_full_to_half(const T1* full, T2* half, const int dims) {
	for (int d1 = 0; d1 < dims; d1++)
		for (int d2 = 0; d2 <= d1; d2++)
			half[(d1*d1 + d1) / 2 + d2] = (T1)full[d1*dims + d2];
}

#ifdef FULL_LEDOIT_WOLF_ESTIMATOR
__global__ static void compute_LW_b2(float* d_weight, float* d_weighted_covar, float* d_o_weighted_LW_b2, int N, int dims) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < N) {
		const int covar_dims = (dims*dims + dims) / 2;
		float weight = d_weight[idx];
		float norm2 = 0;
		for (int d1 = 0; d1 < dims; d1++)
			for (int d2 = 0; d2 <= d1; d2++) {
				float tmp = d_weighted_covar[idx*covar_dims + (d1*d1 + d1) / 2 + d2] - weight * c_covar[(d1*d1 + d1) / 2 + d2];
				if (d1 != d2)
					norm2 += 2 * tmp*tmp;
				else
					norm2 += tmp * tmp;
			}
		d_o_weighted_LW_b2[idx] = norm2;
	}
}
#endif

// computes weights/weighted-mean/weighted-covar for each sample
__global__ static void e_step( // c_mean, c_covar_inv are needed
	float* d_space,
	float* d_o_weight, float* d_o_weighted_space, float* d_o_weighted_covar,
	float trunc_sigma,
	int N, int dims) {

	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < N) {
		float diff[MAX_DIMS];

		for (int d = 0; d < dims; d++)
			diff[d] = d_space[idx*dims + d] - c_mean[d];


		float z_score = 0;
		for (int d1 = 0; d1 < dims; d1++) {
			float tmp = 0;
			for (int d2 = 0; d2 < dims; d2++) {
				if (d1 >= d2)
					tmp += c_covar_inv[(d1*d1 + d1) / 2 + d2] * diff[d2];
				else
					tmp += c_covar_inv[(d2*d2 + d2) / 2 + d1] * diff[d2];
			}
			z_score += tmp * diff[d1];
		}
		z_score = sqrtf(z_score);

		const float weight = z_score < trunc_sigma ? 1 : 0;

		d_o_weight[idx] = weight;

		for (int d = 0; d < dims; d++)
			d_o_weighted_space[idx*dims + d] = weight * d_space[idx*dims + d];

		const int covar_dims = (dims*dims + dims) / 2;
		for (int d1 = 0; d1 < dims; d1++)
			for (int d2 = 0; d2 <= d1; d2++)
				d_o_weighted_covar[idx*covar_dims + (d1*d1 + d1) / 2 + d2] = weight * diff[d1] * diff[d2];

	}
}



int fit_robust_gaussian(
	float* h_space, float* h_io_mean, float* h_io_covar,
	float trunc_sigma, float covar_reg_lambda,
	float* h_o_density, int* used_iters,
	int N, int dims,
	float epsilon, int max_iters) {
	if (dims > MAX_DIMS)
		throw;

	float* d_space;
	float* d_weight;
	float* d_weighted_space;
	float* d_weighted_covar;

	float* ht_weight; // host temp memory for weight. size=1
	float* ht_mean; // host temp memory for mean. size=dims
	float* ht_covar; // host temp memory for mean. size=dims*dims
	float* ht_covar_inv; // host temp memory for mean. size=dims*dims
	double* ht_covar_full; // host temp memory for mean. size=dims*dims
	double* ht_covar_inv_full; // host temp memory for mean. size=dims*dims

	const int dims_covar = (dims*dims + dims) / 2;

	ht_weight = new float[1];
	ht_mean = new float[dims];
	ht_covar = new float[dims_covar];
	ht_covar_inv = new float[dims_covar];
	ht_covar_full = new double[dims*dims];
	ht_covar_inv_full = new double[dims*dims];

	*ht_weight = 0;
	for (int d = 0; d < dims; d++)
		ht_mean[d] = h_io_mean[d];
	for (int d1 = 0; d1 < dims; d1++)
		for (int d2 = 0; d2 <= d1; d2++)
			ht_covar[(d1*d1 + d1) / 2 + d2] = h_io_covar[d1*dims + d2];


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
	cudaMalloc((void**)&d_weighted_covar, (N + N_reduce_sum_ext) * dims_covar * sizeof(float));
	gpuErrchk;

#ifdef FULL_LEDOIT_WOLF_ESTIMATOR
	float* d_weighted_LW_b2;
	float* ht_LW_b2; // host temp memory for LW_b2. size=1
	ht_LW_b2 = new float[1];
	cudaMalloc((void**)&d_weighted_LW_b2, (N + N_reduce_sum_ext) * sizeof(float));
#endif

	if (used_iters != NULL)
		*used_iters = 0;

	// start iteration
	int iter;
	bool result_is_reliable = true;
	for (iter = 0; iter < max_iters; iter++) {

		cudaMemcpyToSymbol(c_mean, ht_mean, dims * sizeof(float));


		// process covar
		covar_half_to_full(ht_covar, ht_covar_full, dims);

		double covar_det; // important to have det double precision, since it is usually very small

		covar_det = determinant(ht_covar_full, dims);
		//printf("covar det = %.20lf\n", covar_det);

		//regularize covar
		if (iter > 0 && covar_reg_lambda > 0) {
#ifdef FULL_LEDOIT_WOLF_ESTIMATOR
			cudaMemcpyToSymbol(c_covar, ht_covar, dims_covar * sizeof(float));
			gpuErrchk;
			compute_LW_b2 << <DIV_CEIL(N, N_THREADS), N_THREADS >> > (d_weight, d_weighted_covar, d_weighted_LW_b2, N, dims);
			gpuErrchk;
			reduce_vector_sum<N_THREADS>(d_weighted_LW_b2, ht_LW_b2, N, 1);
			gpuErrchk;
			(*ht_LW_b2) /= SQR(*ht_weight);

			regularize_covar_LW(ht_covar_full, ht_covar_full, (double)(*ht_LW_b2), dims);
#else
			regularize_covar_LW_given_lambda(ht_covar_full, ht_covar_full, covar_reg_lambda, dims);
#endif
		}
		//for (int d = 0; d < dims; d++)printf("%f ", ht_covar_full[d*dims + d]); printf("\n");
		covar_det = inverse(ht_covar_full, ht_covar_inv_full, dims);
		if (covar_det <= 0) {
			result_is_reliable = false;
			break;
		}

		covar_full_to_half(ht_covar_full, ht_covar, dims);
		covar_full_to_half(ht_covar_inv_full, ht_covar_inv, dims);


		cudaMemcpyToSymbol(c_covar_inv, ht_covar_inv, dims_covar * sizeof(float));
		gpuErrchk;

		// compute displacement
		float prev_density = *ht_weight / N;
		e_step << <DIV_CEIL(N, N_THREADS), N_THREADS >> > (d_space, d_weight, d_weighted_space, d_weighted_covar, trunc_sigma, N, dims);

		// m step
		reduce_vector_sum<N_THREADS>(d_weight, ht_weight, N, 1);
		gpuErrchk;

		if (!isfinite(*ht_weight)) {
			result_is_reliable = false;
			break;
		}

		float density_change = abs(*ht_weight / N - prev_density);

		if (density_change < epsilon) {
			result_is_reliable = true;
			break;
		}

		// update mean and covar only if no converge.
		// otherwise, mean and covar(reged) used in pervious iteration will be returned.

		reduce_vector_sum<N_THREADS>(d_weighted_space, ht_mean, N, dims);
		reduce_vector_sum<N_THREADS>(d_weighted_covar, ht_covar, N, dims_covar);
		gpuErrchk;

		for (int d = 0; d < dims; d++)
			ht_mean[d] /= *ht_weight;

		for (int d1 = 0; d1 < dims; d1++)
			for (int d2 = 0; d2 <= d1; d2++)
				ht_covar[(d1*d1 + d1) / 2 + d2] /= (*ht_weight); // no -1 since it will be regularized later
				//ht_covar[(d1*d1 + d1) / 2 + d2] /= (*ht_weight - 1); // -1 for small sample number correction

		//printf("iter %d, density = %f, change = %f, covar_det = %e\n", iter, *ht_weight / N, density_change, covar_det);
		//for (int d = 0; d < dims; d++)printf("%f ", ht_covar[(d*d + d) / 2 + d]);printf("\n");
	}



	// update confidence & used_iter
	if (result_is_reliable) {
		if (h_o_density != NULL)
			*h_o_density = *ht_weight / N;
		if (used_iters != NULL)
			*used_iters = iter;
		for (int d1 = 0; d1 < dims; d1++)
			for (int d2 = 0; d2 <= d1; d2++) {
				h_io_covar[d1*dims + d2] = ht_covar[(d1*d1 + d1) / 2 + d2];
				h_io_covar[d2*dims + d1] = h_io_covar[d1*dims + d2];
			}
		for (int d = 0; d < dims; d++)
			h_io_mean[d] = ht_mean[d];
	}

	delete[] ht_weight;
	delete[] ht_mean;
	delete[] ht_covar;
	delete[] ht_covar_inv;
	delete[] ht_covar_full;
	delete[] ht_covar_inv_full;

	cudaFree(d_space);
	cudaFree(d_weight);
	cudaFree(d_weighted_space);
	cudaFree(d_weighted_covar);
#ifdef FULL_LEDOIT_WOLF_ESTIMATOR
	cudaFree(d_weighted_LW_b2);
	delete[] ht_LW_b2;
#endif
	gpuErrchk;

	if (result_is_reliable)
		return cudaSuccess;
	else
		return !cudaSuccess;
}

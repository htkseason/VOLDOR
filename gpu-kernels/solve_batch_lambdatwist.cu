#include "utils.h"
#include "gpu_kernels.h"
#include "rodrigues.h"
#include "../lambdatwist/lambdatwist_p4p.h"

#define N_THREADS 32

__constant__ static float _fx, _fy, _cx, _cy;


__global__ static void solve(float* d_p2s, float* d_p3s, float* d_rvecs, float* d_tvecs, curandState* d_rand_states, int N_pts, int N_poses) {

	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < N_poses) {
		int i1 = curand_uniform(&d_rand_states[idx])*N_pts;
		int i2 = curand_uniform(&d_rand_states[idx])*N_pts;
		int i3 = curand_uniform(&d_rand_states[idx])*N_pts;
		int i4 = curand_uniform(&d_rand_states[idx])*N_pts;
		float R[3][3], t[3];

		bool success = lambdatwist_p4p<float, float, 5>(
			&d_p2s[i1 * 2], &d_p2s[i2 * 2], &d_p2s[i3 * 2], &d_p2s[i4 * 2],
			&d_p3s[i1 * 3], &d_p3s[i2 * 3], &d_p3s[i3 * 3], &d_p3s[i4 * 3],
			_fx, _fy, _cx, _cy,
			R, t);


		if (!success) {
			d_rvecs[idx * 3 + 0] = CUDART_NAN_F; d_rvecs[idx * 3 + 1] = CUDART_NAN_F; d_rvecs[idx * 3 + 2] = CUDART_NAN_F;
			d_tvecs[idx * 3 + 0] = CUDART_NAN_F; d_tvecs[idx * 3 + 1] = CUDART_NAN_F; d_tvecs[idx * 3 + 2] = CUDART_NAN_F;
			return; // false
		}
		
		d_tvecs[idx * 3 + 0] = t[0];
		d_tvecs[idx * 3 + 1] = t[1];
		d_tvecs[idx * 3 + 2] = t[2];
		rodrigues(R, &d_rvecs[idx * 3]);
		
		
	}
}

__global__ static void init_rand_states(curandState* d_rand_states, int N) {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	if (idx < N)
		curand_init(RAND_SEED, idx, 0, &d_rand_states[idx]);
}


int solve_batch_p3p_lambdatwist_gpu(float* h_p3s, float* h_p2s, float* h_o_rvecs, float* h_o_tvecs, float* h_K, int N_pts, int N_poses) {

	// copy K to gpu constant memory
	static float cache_symbols[4] = { 0 };
	if (h_K) {
		CUDA_UPDATE_SYMBOL_IF_CHANGED(h_K[0], cache_symbols[0], _fx);
		CUDA_UPDATE_SYMBOL_IF_CHANGED(h_K[2], cache_symbols[2], _cx);
		CUDA_UPDATE_SYMBOL_IF_CHANGED(h_K[4], cache_symbols[1], _fy);
		CUDA_UPDATE_SYMBOL_IF_CHANGED(h_K[5], cache_symbols[3], _cy);
	}

	// allocate and copy pts2+pts3
	float* d_p2s;
	float* d_p3s;
	cudaMalloc((void**)&d_p2s, N_pts * 2 * sizeof(float));
	cudaMalloc((void**)&d_p3s, N_pts * 3 * sizeof(float));
	cudaMemcpy(d_p2s, h_p2s, N_pts * 2 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_p3s, h_p3s, N_pts * 3 * sizeof(float), cudaMemcpyHostToDevice);
	gpuErrchk;

	// allocate device memory for R,t
	float* d_rvecs;
	float* d_tvecs;
	cudaMalloc((void**)&d_rvecs, N_poses * 3 * sizeof(float));
	cudaMalloc((void**)&d_tvecs, N_poses * 3 * sizeof(float));
	gpuErrchk;

	// init rand seeds
	curandState* d_rand_states;
	cudaMalloc((void**)&d_rand_states, N_poses * sizeof(curandState));
	init_rand_states << <DIV_CEIL(N_poses, N_THREADS), N_THREADS >> > (d_rand_states, N_poses);

	// solve batch ap3p
	float fx = h_K[0], cx = h_K[2], fy = h_K[4], cy = h_K[5];
	solve << <DIV_CEIL(N_poses, N_THREADS), N_THREADS >> > (d_p2s, d_p3s, d_rvecs, d_tvecs, d_rand_states, N_pts, N_poses);
	gpuErrchk;

	// copy back R,t
	cudaMemcpy(h_o_rvecs, d_rvecs, N_poses * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_o_tvecs, d_tvecs, N_poses * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	gpuErrchk;

	// free all
	cudaFree(d_rvecs);
	cudaFree(d_tvecs);
	cudaFree(d_p2s);
	cudaFree(d_p3s);
	cudaFree(d_rand_states);
	gpuErrchk;

	return cudaSuccess;
}

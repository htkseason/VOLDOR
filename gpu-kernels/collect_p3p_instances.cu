#include "utils.h"
#include "gpu_kernels.h"
#include "gmat.h"


#define RAND_SEED 233

#define BLOCK_WIDTH 16

#define MAX_FRAMES 16

__constant__ static float _K4[4]; // _fx,_cx,_fy,_cy
__constant__ static float _K4_inv[4]; // 1/_fx, -_cx/_fx, 1/_fy, -_cy/_fy
__constant__ static float _Rs[MAX_FRAMES][3][3];
__constant__ static float _ts[MAX_FRAMES][3];

__constant__ static int _N, _w, _h;



__constant__ static GMatf2 _d_flows;
__constant__ static GMatf _d_depth;
__constant__ static GMatf _d_rigidnesses;
__constant__ static GMatf _d_rigidnesses_sum;
__constant__ static GMatf2 _d_p2_map;
__constant__ static GMatf3 _d_p3_map;


static GMatf2 d_flows;
static GMatf d_depth;
static GMatf d_rigidnesses;
static GMatf d_rigidnesses_sum;
static GMatf2 d_p2_map;
static GMatf3 d_p3_map;

__device__ __inline__ static void proj_p2_to_p3(float px, float py, float depth, float& ox, float& oy, float& oz) {
	ox = (_K4_inv[0] * px + _K4_inv[1]) * depth;
	oy = (_K4_inv[2] * py + _K4_inv[3]) * depth;
	oz = depth;
}

__device__ __inline__ static void proj_p3_to_p2(float ox, float oy, float oz, float& px, float& py) {
	px = (_K4[0] * ox + _K4[1] * oz) / oz;
	py = (_K4[2] * oy + _K4[3] * oz) / oz;
}

__device__ __inline__ static void trans_p3_across_frame(float& ox, float& oy, float& oz, int f) {
	float ox_temp = ox * _Rs[f][0][0] + oy * _Rs[f][0][1] + oz * _Rs[f][0][2];
	float oy_temp = ox * _Rs[f][1][0] + oy * _Rs[f][1][1] + oz * _Rs[f][1][2];
	float oz_temp = ox * _Rs[f][2][0] + oy * _Rs[f][2][1] + oz * _Rs[f][2][2];
	ox = ox_temp + _ts[f][0];
	oy = oy_temp + _ts[f][1];
	oz = oz_temp + _ts[f][2];
}


__global__ static void compute_rigidnesses_sum() {
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	if (x < _w && y < _h) {
		_d_rigidnesses_sum.at(x, y) = 0;
		for (int i = 0; i < _N; i++)
			_d_rigidnesses_sum.at(x, y) += _d_rigidnesses.at(x, y, i);

	}

}


__global__ static void compute_p3p_map(
	const int active_idx,
	const float rigidness_thresh,
	const float rigidness_sum_thresh,
	const float sample_min_depth,
	const float sample_max_depth,
	const int max_trace_on_flow) {

	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	
	if (x < _w && y < _h) {
		_d_p2_map.at(x, y) = make_float2(CUDART_NAN_F, CUDART_NAN_F);
		_d_p3_map.at(x, y) = make_float3(CUDART_NAN_F, CUDART_NAN_F, CUDART_NAN_F);

		float depth = _d_depth.at(x, y);
		if (depth < sample_min_depth || (sample_max_depth > 0 && depth > sample_max_depth))
			return;
		if (_d_rigidnesses_sum.at(x, y) < rigidness_sum_thresh &&
			rigidness_sum_thresh > _N + 1)
			return;

		int n_trace_on_flow = 0;
		float trace_product = 1;
		for (int i = active_idx; i >= (max_trace_on_flow > 0 ? max(0, active_idx - max_trace_on_flow + 1) : 0); i--) {
			trace_product *= _d_rigidnesses.at(x, y, i);
			if (trace_product > rigidness_thresh)
				n_trace_on_flow++;
			else
				break;
		}

		if (n_trace_on_flow <= 0)
			return;


		bool p2_trace_out_boundary = false;
		float px, py;
		float ox, oy, oz;
		proj_p2_to_p3(x, y, depth, ox, oy, oz);

		for (int i = 0; i <= active_idx; i++) {
			// tracing on flow
			if (i >= active_idx - n_trace_on_flow + 1) {
				// convert p3 to p2 and start tracing on flow
				if (i == active_idx - n_trace_on_flow + 1) {
					proj_p3_to_p2(ox, oy, oz, px, py);
				}

				if (px > 0 && px < _w && py > 0 && py < _h) {
					float2 d2 = _d_flows.at_tex(px, py, i);
					px += d2.x;
					py += d2.y;
				}
				else {
					p2_trace_out_boundary = true;
					break;
				}
			}
			// trans p3 to next frame
			if (i < active_idx) {
				trans_p3_across_frame(ox, oy, oz, i);
			}
		}


		// check depth w.r.t. frame active_idx
		if (!p2_trace_out_boundary && oz > sample_min_depth && (sample_max_depth <= 0 || oz < sample_max_depth)) {
			_d_p2_map.at(x, y) = make_float2(px, py);
			_d_p3_map.at(x, y) = make_float3(ox, oy, oz);
		}


	}

}

int collect_p3p_instances(
	float* h_flows[], float* h_rigidnesses[],
	float* h_depth,
	float* h_K, float* h_Rs[], float* h_ts[],
	float* h_o_p2_map, float* h_o_p3_map,
	int N, int w, int h,
	int active_idx,
	float rigidness_thresh,
	float rigidness_sum_thresh,
	float sample_min_depth,
	float sample_max_depth,
	int max_trace_on_flow) {

	// for pixel-wise op
	const dim3 block_size(BLOCK_WIDTH, BLOCK_WIDTH);
	const dim3 grid_size(DIV_CEIL(w, BLOCK_WIDTH), DIV_CEIL(h, BLOCK_WIDTH));

	// copy params to constant memory
	static float cache_symbols[3] = { 0 };
	CUDA_UPDATE_SYMBOL_IF_CHANGED(N, cache_symbols[0], _N);
	CUDA_UPDATE_SYMBOL_IF_CHANGED(w, cache_symbols[1], _w);
	CUDA_UPDATE_SYMBOL_IF_CHANGED(h, cache_symbols[2], _h);


	// copy camera info to constant memory
	// b_xxx stands for temp buffer
	if (h_K) {
		float b_K4[4]{ h_K[0] , h_K[2], h_K[4],h_K[5] };
		float b_K4_inv[4]{ 1.f / h_K[0], -h_K[2] / h_K[0], 1.f / h_K[4], -h_K[5] / h_K[4] };
		cudaMemcpyToSymbol(_K4, b_K4, 4 * sizeof(float));
		cudaMemcpyToSymbol(_K4_inv, b_K4_inv, 4 * sizeof(float));
	}


	if (h_Rs) {
		float b_R[MAX_FRAMES][3][3];
		for (int f = 0; f < N; f++)
			memcpy(b_R[f], h_Rs[f], 9 * sizeof(float));
		cudaMemcpyToSymbol(_Rs, b_R, N * 9 * sizeof(float));
		gpuErrchk;
	}

	if (h_ts) {
		float b_t[MAX_FRAMES][3];
		for (int f = 0; f < N; f++)
			memcpy(b_t[f], h_ts[f], 3 * sizeof(float));
		cudaMemcpyToSymbol(_ts, b_t, N * 3 * sizeof(float));
		gpuErrchk;
	}

	// copy flow to device
	if (d_flows.create(w, h, N, true)) {
		d_flows.bind_tex();
		cudaMemcpyToSymbol(_d_flows, &d_flows, sizeof(GMatf2));
	}
	if (h_flows) {	
		for (int f = 0; f < N; f++)
			d_flows.copy_from_host((float2*)h_flows[f], make_cudaPos(0, 0, f), w, h, 1);
		gpuErrchk;
	}

	if (d_rigidnesses_sum.create(w, h, 1))
		cudaMemcpyToSymbol(_d_rigidnesses_sum, &d_rigidnesses_sum, sizeof(GMatf));
	gpuErrchk;

	// copy rigidnesses
	if (d_rigidnesses.create(w, h, N, true))
		cudaMemcpyToSymbol(_d_rigidnesses, &d_rigidnesses, sizeof(GMatf));
	if (h_rigidnesses) {
		for (int f = 0; f < N; f++)
			d_rigidnesses.copy_from_host(h_rigidnesses[f], make_cudaPos(0, 0, f), w, h, 1);
		compute_rigidnesses_sum << <grid_size, block_size >> > ();
	}
	gpuErrchk;

	// copy depth to device
	if (d_depth.create(w, h, 1))
		cudaMemcpyToSymbol(_d_depth, &d_depth, sizeof(GMatf));
	if (h_depth) {
		d_depth.copy_from_host(h_depth, make_cudaPos(0, 0, 0), w, h, 1);
	}
	gpuErrchk;


	if (d_p2_map.create(w, h, 1))
		cudaMemcpyToSymbol(_d_p2_map, &d_p2_map, sizeof(GMatf2));
	gpuErrchk;

	if (d_p3_map.create(w, h, 1))
		cudaMemcpyToSymbol(_d_p3_map, &d_p3_map, sizeof(GMatf3));
	gpuErrchk;


	compute_p3p_map << <grid_size, block_size >> > (active_idx, rigidness_thresh, rigidness_sum_thresh, sample_min_depth, sample_max_depth, max_trace_on_flow);
	gpuErrchk;

	if (h_o_p2_map)
		d_p2_map.copy_to_host((float2*)h_o_p2_map, make_cudaPos(0, 0, 0), w, h, 1);
	if (h_o_p3_map)
		d_p3_map.copy_to_host((float3*)h_o_p3_map, make_cudaPos(0, 0, 0), w, h, 1);
	gpuErrchk;

	return cudaSuccess;
}

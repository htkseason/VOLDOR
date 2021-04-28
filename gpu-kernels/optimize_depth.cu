#include "utils.h"
#include "gpu_kernels.h"
#include "residual_model.h"
#include "fb_smooth.h"
#include "gmat.h"


#define RAND_SEED 233

#define PROPAGATE_L2R 0
#define PROPAGATE_T2B 1
#define PROPAGATE_R2L 2
#define PROPAGATE_B2T 3

#define MAXIMUM_DEPTH 1e5f

#define BLOCK_WIDTH 16
#define CHAIN_WIDTH 64 // we do not want this too large..

#define MAX_FRAMES 16
#define MAX_DISP_FRAMES 16


__constant__ static float _K4[4]; // _fx,_cx,_fy,_cy
__constant__ static float _K4_inv[4]; // 1/_fx, -_cx/_fx, 1/_fy, -_cy/_fy
__constant__ static float _Rs[MAX_FRAMES][3][3];
__constant__ static float _ts[MAX_FRAMES][3];
__constant__ static float _dp_Rs[MAX_DISP_FRAMES][3][3];
__constant__ static float _dp_ts[MAX_DISP_FRAMES][3];
__constant__ static int _N, _N_dp, _w, _h;
__constant__ static float _abs_resize_factor;
__constant__ static float _basefocal;
__constant__ static float _lambda, _omega, _delta, _disp_delta;
__constant__ static float _range_factor; // sampling = 1/(factor*rnd + (1/max))

__constant__ static GMatRnd _d_rand_states;
__constant__ static GMatf2 _d_flows;
__constant__ static GMatf _d_depth_priors;
__constant__ static GMatf _d_depth_prior_pconfs;
__constant__ static GMatf _d_depth_prior_confs;
__constant__ static GMatf _d_depth;
__constant__ static GMatf _d_rigidnesses;
__constant__ static GMatf _d_cost_map;

static GMatRnd d_rand_states;
static GMatf2 d_flows;
static GMatf d_depth_priors;
static GMatf d_depth_prior_pconfs;
static GMatf d_depth_prior_confs;
static GMatf d_depth;
static GMatf d_rigidnesses;
static GMatf d_cost_map;

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

__device__ __inline__ static void trans_p3_across_dp(float& ox, float& oy, float& oz, int f) {
	float ox_temp = ox * _dp_Rs[f][0][0] + oy * _dp_Rs[f][0][1] + oz * _dp_Rs[f][0][2];
	float oy_temp = ox * _dp_Rs[f][1][0] + oy * _dp_Rs[f][1][1] + oz * _dp_Rs[f][1][2];
	float oz_temp = ox * _dp_Rs[f][2][0] + oy * _dp_Rs[f][2][1] + oz * _dp_Rs[f][2][2];
	ox = ox_temp + _dp_ts[f][0];
	oy = oy_temp + _dp_ts[f][1];
	oz = oz_temp + _dp_ts[f][2];
}


__global__ static void update_rigidnesses() {
	const int x = blockDim.x*blockIdx.x + threadIdx.x;
	const int y = blockDim.y*blockIdx.y + threadIdx.y;

	if (x < _w && y < _h) {

		float ox, oy, oz;	// 3d point coordinate
		float px1, py1;		// 2d projection of 3d point in frame f
		float px2, py2;		// 2d projection of 3d point in frame f+1
		float dx1, dy1;		// pixel displacement of depth flow
		float2 d2;		// pixel displacement of observed flow

		proj_p2_to_p3(x, y, _d_depth.at(x, y), ox, oy, oz);	// init 3d point coordinate with regard to frame 0
		px1 = x, py1 = y; // init 3d point projection in frame 0

		for (int f = 0; f < _N; f++) {
			trans_p3_across_frame(ox, oy, oz, f);	// trans 3d point to frame f+1
			proj_p3_to_p2(ox, oy, oz, px2, py2);	// get 2d projection of 3d point in frame f+1

			if (oz > 0 && // check if depth is positive
				px1 >= 0 && px1 < _w && py1 >= 0 && py1 < _h) {
				d2 = _d_flows.at_tex(px1, py1, f); // get observed 
				dx1 = px2 - px1, dy1 = py2 - py1;	// compute rigid flow
				px1 = px2, py1 = py2;
				_d_rigidnesses.at(x, y, f) = fun_rigidness(dx1, dy1, d2.x, d2.y, _lambda, _abs_resize_factor);
			}
			else {
				_d_rigidnesses.at(x, y, f) = 0;
				//for (int f2 = f; f2 < _N; f2++)
					//o_rigidnesses.at(x, y, f2) = 0;
				//return;
			}
		}


		for (int f = 0; f < _N_dp; f++) {
			proj_p2_to_p3(x, y, _d_depth.at(x, y), ox, oy, oz);	// init 3d point coordinate with regard to frame 0
			trans_p3_across_dp(ox, oy, oz, f);	// trans 3d point to dp frame f
			proj_p3_to_p2(ox, oy, oz, px1, py1);	// get 2d projection of 3d point

			if (oz > 0 && // check if depth is positive
				px1 >= 0 && px1 < _w && py1 >= 0 && py1 < _h) {

				float target_depth = _d_depth_priors.at_tex(px1, py1, f);

				if (target_depth > 0)
					_d_depth_prior_confs.at(x, y, f) = fun_depth_rigidness(oz, target_depth, _basefocal, _omega, _abs_resize_factor);
			}
			else {
				_d_depth_prior_confs.at(x, y, f) = 0;
			}
		}

	}
}

__device__ static float compute_pixel_cost(const int px, const int py, const float depth) {
	float cost_sum = 0;
	float rigidness_weight_sum = 0;

	float ox, oy, oz;	// 3d point coordinate
	float px1, py1;		// 2d projection of 3d point in frame f
	float px2, py2;		// 2d projection of 3d point in frame f+1
	float dx1, dy1;		// pixel displacement of depth flow
	float2 d2;		// pixel displacement of observed flow

	proj_p2_to_p3(px, py, depth, ox, oy, oz);	// init 3d point coordinate with regard to frame 0
	px1 = px, py1 = py; // init 3d point projection in frame 0

	for (int f = 0; f < _N; f++) {
		trans_p3_across_frame(ox, oy, oz, f);	// trans 3d point to frame f+1
		proj_p3_to_p2(ox, oy, oz, px2, py2);	// get 2d projection of 3d point in frame f+1

		if (oz > 0 && // check if depth is positive
			px1 >= 0 && px1 < _w && py1 >= 0 && py1 < _h) {
			d2 = _d_flows.at_tex(px1, py1, f); // get observed flow	
			dx1 = px2 - px1, dy1 = py2 - py1;	// compute rigid flow
			px1 = px2, py1 = py2;

			fun_cost(dx1, dy1, d2.x, d2.y, _d_rigidnesses.at(px, py, f),
				cost_sum, rigidness_weight_sum, _lambda, _abs_resize_factor);
		}
		else {
			continue;
		}
	}


	for (int f = 0; f < _N_dp; f++) {
		proj_p2_to_p3(px, py, depth, ox, oy, oz);	// init 3d point coordinate with regard to frame 0
		trans_p3_across_dp(ox, oy, oz, f);	// trans 3d point to dp frame f
		proj_p3_to_p2(ox, oy, oz, px1, py1);	// get 2d projection of 3d point

		if (oz > 0 &&
			px1 >= 0 && px1 < _w && py1 >= 0 && py1 < _h) {

			float target_depth = _d_depth_priors.at_tex(px1, py1, f);
			float target_depth_pconf = _d_depth_prior_pconfs.at_tex(px1, py1, f);
			float target_depth_conf = _d_depth_prior_confs.at_tex(px1, py1, f);

			if (target_depth > 0) {
				if (_disp_delta > 0 && f == 0)
					fun_depth_cost(oz, target_depth, _basefocal, target_depth_pconf*target_depth_conf*_disp_delta, cost_sum, rigidness_weight_sum, _omega, _abs_resize_factor);
				else
					fun_depth_cost(oz, target_depth, _basefocal, target_depth_pconf*target_depth_conf*_delta, cost_sum, rigidness_weight_sum, _omega, _abs_resize_factor);
			}
		}
	}


	if (rigidness_weight_sum == 0) // prevent both cost_sum=0
		return INFINITY;

	return cost_sum / fmaxf(rigidness_weight_sum, ZDE);
}

// compute cost of depth_new. if less than given reporj err, replace o_depth with depth_new
__device__ __inline__ static void replace_if_better_depth(int px, int py, float depth_new, float& o_depth, float& io_cost) {
	float cost = compute_pixel_cost(px, py, depth_new);
	if (cost < io_cost) {
		o_depth = depth_new;
		io_cost = cost;
	}
}

template<int propagate_direction>
__global__ static void optimize_depth_with_global_propagation_inplace(int step) {
	const int tx = blockDim.x*blockIdx.x + threadIdx.x;
	const int ty = blockDim.y*blockIdx.y + threadIdx.y;
	if (tx < _w && ty < _h) {
		if (propagate_direction == PROPAGATE_L2R) {
			for (int x = 1; x < _w; x += step)
				replace_if_better_depth(x, ty,
					_d_depth.at(x - 1, ty), _d_depth.at(x, ty), _d_cost_map.at(x, ty));
		}
		else if (propagate_direction == PROPAGATE_R2L) {
			for (int x = _w - 2; x >= 0; x -= step)
				replace_if_better_depth(x, ty,
					_d_depth.at(x + 1, ty), _d_depth.at(x, ty), _d_cost_map.at(x, ty));
		}
		else if (propagate_direction == PROPAGATE_T2B) {
			for (int y = 1; y < _h; y += step)
				replace_if_better_depth(tx, y,
					_d_depth.at(tx, y - 1), _d_depth.at(tx, y), _d_cost_map.at(tx, y));
		}
		else if (propagate_direction == PROPAGATE_B2T) {
			for (int y = _h - 2; y >= 0; y -= step)
				replace_if_better_depth(tx, y,
					_d_depth.at(tx, y + 1), _d_depth.at(tx, y), _d_cost_map.at(tx, y));
		}
	}
}

template<int propagate_direction>
__global__ static void optimize_depth_with_local_propagation_inplace(int width) {
	const int tx = blockDim.x*blockIdx.x + threadIdx.x;
	const int ty = blockDim.y*blockIdx.y + threadIdx.y;
	if (tx < _w && ty < _h) {
		if (propagate_direction == PROPAGATE_L2R) {
			int px = tx * width;
			for (int x = max(1, px + 1); x < min(_w, px + width); x++)
				replace_if_better_depth(x, ty,
					_d_depth.at(x - 1, ty), _d_depth.at(x, ty), _d_cost_map.at(x, ty));
		}
		else if (propagate_direction == PROPAGATE_R2L) {
			int px = tx * width;
			for (int x = min(_w - 2, px + width - 2); x >= max(0, px); x--)
				replace_if_better_depth(x, ty,
					_d_depth.at(x + 1, ty), _d_depth.at(x, ty), _d_cost_map.at(x, ty));
		}
		else if (propagate_direction == PROPAGATE_T2B) {
			int py = ty * width;
			for (int y = max(1, py + 1); y < min(_h, py + width); y++)
				replace_if_better_depth(tx, y,
					_d_depth.at(tx, y - 1), _d_depth.at(tx, y), _d_cost_map.at(tx, y));
		}
		else if (propagate_direction == PROPAGATE_B2T) {
			int py = ty * width;
			for (int y = min(_h - 2, py + width - 2); y >= max(0, py); y--)
				replace_if_better_depth(tx, y,
					_d_depth.at(tx, y + 1), _d_depth.at(tx, y), _d_cost_map.at(tx, y));
		}
	}
}

__global__ static void optimize_depth_with_rand_inplace() {
	const int x = blockDim.x*blockIdx.x + threadIdx.x;
	const int y = blockDim.y*blockIdx.y + threadIdx.y;
	if (x < _w && y < _h) {
		float depth_rnd = 1.0f / (_range_factor*curand_uniform(&_d_rand_states.at(x, y)) + (1.0f / MAXIMUM_DEPTH));
		replace_if_better_depth(x, y,
			depth_rnd, _d_depth.at(x, y), _d_cost_map.at(x, y));
	}
}

__global__ static void compute_cost_map() {
	const int x = blockDim.x*blockIdx.x + threadIdx.x;
	const int y = blockDim.y*blockIdx.y + threadIdx.y;
	if (x < _w && y < _h)
		_d_cost_map.at(x, y) = compute_pixel_cost(x, y, _d_depth.at(x, y));
}

__global__ static void init_rand_states() {
	const int x = blockDim.x*blockIdx.x + threadIdx.x;
	const int y = blockDim.y*blockIdx.y + threadIdx.y;
	if (x < _w && y < _h)
		curand_init(RAND_SEED, y*_w + x, 0, &_d_rand_states.at(x, y));
}

int optimize_depth_gpu(
	float* h_flows[],
	float* h_rigidnesses[], float* h_o_rigidnesses[],
	float* h_depth_priors[], float* h_depth_prior_pconfs[],
	float* h_depth_prior_confs[], float* h_o_depth_prior_confs[],
	float* h_depth, float* h_o_depth,
	float* h_K, float* h_Rs[], float* h_ts[],
	float* h_dp_Rs[], float* h_dp_ts[],
	float abs_resize_factor,
	int N, int N_dp, int w, int h, float basefocal,
	int n_rand_samples, int global_prop_step, int local_prop_width,
	float lambda, float omega, float disp_delta, float delta,
	bool fb_smooth, float s0_ems_prob, float no_change_prob,
	float range_factor,
	bool update_rigidness_only) {

	// for pixel-wise op
	const dim3 block_size(BLOCK_WIDTH, BLOCK_WIDTH);
	const dim3 grid_size(DIV_CEIL(w, BLOCK_WIDTH), DIV_CEIL(h, BLOCK_WIDTH));

	// for global propagation
	const dim3 block_size_row_chain(1, CHAIN_WIDTH);
	const dim3 grid_size_row_chain(1, DIV_CEIL(h, CHAIN_WIDTH));

	const dim3 block_size_col_chain(CHAIN_WIDTH, 1);
	const dim3 grid_size_col_chain(DIV_CEIL(w, CHAIN_WIDTH), 1);

	// for local propagation
	const dim3 block_size_row_step(BLOCK_WIDTH, BLOCK_WIDTH);
	const dim3 grid_size_row_step(DIV_CEIL(w, BLOCK_WIDTH*local_prop_width), DIV_CEIL(h, BLOCK_WIDTH));

	const dim3 block_size_col_step(BLOCK_WIDTH, BLOCK_WIDTH);
	const dim3 grid_size_col_step(DIV_CEIL(w, BLOCK_WIDTH), DIV_CEIL(h, BLOCK_WIDTH*local_prop_width));


	// copy hyper-params to constant memory
	static float cache_symbols[11] = { 0 };
	CUDA_UPDATE_SYMBOL_IF_CHANGED(N, cache_symbols[0], _N);
	CUDA_UPDATE_SYMBOL_IF_CHANGED(N_dp, cache_symbols[1], _N_dp);
	CUDA_UPDATE_SYMBOL_IF_CHANGED(w, cache_symbols[2], _w);
	CUDA_UPDATE_SYMBOL_IF_CHANGED(h, cache_symbols[3], _h);
	CUDA_UPDATE_SYMBOL_IF_CHANGED(lambda, cache_symbols[4], _lambda);
	CUDA_UPDATE_SYMBOL_IF_CHANGED(abs_resize_factor, cache_symbols[5], _abs_resize_factor);
	CUDA_UPDATE_SYMBOL_IF_CHANGED(basefocal, cache_symbols[6], _basefocal);
	CUDA_UPDATE_SYMBOL_IF_CHANGED(omega, cache_symbols[7], _omega);
	CUDA_UPDATE_SYMBOL_IF_CHANGED(disp_delta, cache_symbols[8], _disp_delta);
	CUDA_UPDATE_SYMBOL_IF_CHANGED(delta, cache_symbols[9], _delta);
	CUDA_UPDATE_SYMBOL_IF_CHANGED(range_factor, cache_symbols[10], _range_factor);


	// copy camera info to constant memory
	// b_xxx stands for temp buffer
	if (h_K) {
		float b_K4[4]{ h_K[0] , h_K[2], h_K[4],h_K[5] };
		float b_K4_inv[4]{ 1.f / h_K[0], -h_K[2] / h_K[0], 1.f / h_K[4], -h_K[5] / h_K[4] };
		cudaMemcpyToSymbol(_K4, b_K4, 4 * sizeof(float));
		cudaMemcpyToSymbol(_K4_inv, b_K4_inv, 4 * sizeof(float));
	}

	//buffer
	float b_R[MAX_FRAMES > MAX_DISP_FRAMES ? MAX_FRAMES : MAX_DISP_FRAMES][3][3];
	float b_t[MAX_FRAMES > MAX_DISP_FRAMES ? MAX_FRAMES : MAX_DISP_FRAMES][3];


	// init rand states
	if (d_rand_states.create(w, h, 1)) {
		cudaMemcpyToSymbol(_d_rand_states, &d_rand_states, sizeof(GMatRnd));
		init_rand_states << <grid_size, block_size >> > ();
	}
	gpuErrchk;

	// allocate for cost map
	if (d_cost_map.create(w, h, 1))
		cudaMemcpyToSymbol(_d_cost_map, &d_cost_map, sizeof(GMatf));
	gpuErrchk;

	// copy depth to device
	if (d_depth.create(w, h, 1))
		cudaMemcpyToSymbol(_d_depth, &d_depth, sizeof(GMatf));
	if (h_depth) {
		d_depth.copy_from_host(h_depth, make_cudaPos(0, 0, 0), w, h, 1);
	}
	gpuErrchk;

	if (N > 0) {
		if (h_Rs) {
			for (int f = 0; f < N; f++)
				memcpy(b_R[f], h_Rs[f], 9 * sizeof(float));
			cudaMemcpyToSymbol(_Rs, b_R, N * 9 * sizeof(float));
			gpuErrchk;
		}

		if (h_ts) {
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
		}
		gpuErrchk;

		// copy rigidnesses
		if (d_rigidnesses.create(w, h, N, true))
			cudaMemcpyToSymbol(_d_rigidnesses, &d_rigidnesses, sizeof(GMatf));
		if (h_rigidnesses) {
			for (int f = 0; f < N; f++)
				d_rigidnesses.copy_from_host(h_rigidnesses[f], make_cudaPos(0, 0, f), w, h, 1);
		}
		gpuErrchk;
	}


	if (N_dp > 0) {
		if (h_dp_Rs) {
			for (int f = 0; f < N_dp; f++)
				memcpy(b_R[f], h_dp_Rs[f], 9 * sizeof(float));
			cudaMemcpyToSymbol(_dp_Rs, b_R, N_dp * 9 * sizeof(float));
			gpuErrchk;
		}

		if (h_dp_ts) {
			for (int f = 0; f < N_dp; f++)
				memcpy(b_t[f], h_dp_ts[f], 3 * sizeof(float));
			cudaMemcpyToSymbol(_dp_ts, b_t, N_dp * 3 * sizeof(float));
			gpuErrchk;
		}


		if (d_depth_priors.create(w, h, N_dp, true)) {
			d_depth_priors.bind_tex();
			cudaMemcpyToSymbol(_d_depth_priors, &d_depth_priors, sizeof(GMatf));
		}
		if (h_depth_priors) {
			for (int f = 0; f < N_dp; f++)
				d_depth_priors.copy_from_host(h_depth_priors[f], make_cudaPos(0, 0, f), w, h, 1);
		}
		gpuErrchk;

		if (d_depth_prior_pconfs.create(w, h, N_dp, true)) {
			d_depth_prior_pconfs.bind_tex();
			cudaMemcpyToSymbol(_d_depth_prior_pconfs, &d_depth_prior_pconfs, sizeof(GMatf));
		}
		if (h_depth_prior_pconfs) {
			for (int f = 0; f < N_dp; f++)
				d_depth_prior_pconfs.copy_from_host(h_depth_prior_pconfs[f], make_cudaPos(0, 0, f), w, h, 1);
		}
		gpuErrchk;

		if (d_depth_prior_confs.create(w, h, N_dp, true)) {
			d_depth_prior_confs.bind_tex();
			cudaMemcpyToSymbol(_d_depth_prior_confs, &d_depth_prior_confs, sizeof(GMatf));
		}
		if (h_depth_prior_confs) {
			for (int f = 0; f < N_dp; f++)
				d_depth_prior_confs.copy_from_host(h_depth_prior_confs[f], make_cudaPos(0, 0, f), w, h, 1);
		}
		gpuErrchk;
	}


	if (!update_rigidness_only) {
		if (fb_smooth) {
			if (N > 0)
				fb_smooth_batch_inplace(d_rigidnesses, s0_ems_prob, no_change_prob, N, w, h);
			if (N_dp > 0)
				fb_smooth_batch_inplace(d_depth_prior_confs, s0_ems_prob, no_change_prob, N_dp, w, h);
			gpuErrchk;
		}

		// init cost map
		compute_cost_map << <grid_size, block_size >> > ();
		gpuErrchk;

		for (int iter = 0; iter < n_rand_samples; iter++) {
			optimize_depth_with_rand_inplace << <grid_size, block_size >> > ();
			gpuErrchk;
		}

		if (global_prop_step > 0) {
			optimize_depth_with_global_propagation_inplace <PROPAGATE_L2R> << <grid_size_row_chain, block_size_row_chain >> > (global_prop_step);
			optimize_depth_with_global_propagation_inplace <PROPAGATE_B2T> << <grid_size_col_chain, block_size_col_chain >> > (global_prop_step);
			optimize_depth_with_global_propagation_inplace <PROPAGATE_R2L> << <grid_size_row_chain, block_size_row_chain >> > (global_prop_step);
			optimize_depth_with_global_propagation_inplace <PROPAGATE_T2B> << <grid_size_col_chain, block_size_col_chain >> > (global_prop_step);
		}
		if (local_prop_width > 0) {
			optimize_depth_with_local_propagation_inplace<PROPAGATE_L2R> << <grid_size_row_step, block_size_row_step >> > (local_prop_width);
			optimize_depth_with_local_propagation_inplace<PROPAGATE_B2T> << <grid_size_col_step, block_size_col_step >> > (local_prop_width);
			optimize_depth_with_local_propagation_inplace<PROPAGATE_R2L> << <grid_size_row_step, block_size_row_step >> > (local_prop_width);
			optimize_depth_with_local_propagation_inplace<PROPAGATE_T2B> << <grid_size_col_step, block_size_col_step >> > (local_prop_width);
		}
	}

	update_rigidnesses << < grid_size, block_size >> > ();
	gpuErrchk;

	if (h_o_depth)
		d_depth.copy_to_host(h_o_depth, make_cudaPos(0, 0, 0), w, h, 1);

	if (h_o_rigidnesses) {
		for (int f = 0; f < N; f++)
			d_rigidnesses.copy_to_host(h_o_rigidnesses[f], make_cudaPos(0, 0, f), w, h, 1);
	}

	if (h_o_depth_prior_confs) {
		for (int f = 0; f < N_dp; f++)
			d_depth_prior_confs.copy_to_host(h_o_depth_prior_confs[f], make_cudaPos(0, 0, f), w, h, 1);
	}
	gpuErrchk;


	// cudaFree(d_rand_states);
	// cudaFree(d_flows);
	// cudaFree(d_depth);
	// cudaFree(d_rigidnesses);
	// cudaFree(d_cost_map);
	// gpuErrchk;

	return cudaSuccess;
}

#pragma once
#include "utils.h"
#include "gmat.h"

#define FB_MSG_L2R 0
#define FB_MSG_T2B 1
#define FB_MSG_R2L 2
#define FB_MSG_B2T 3
#define FB_POSTERIOR 4

#define FB_BLOCK_WIDTH 16
#define FB_CHAIN_WIDTH 128 // we do not want this too large..

static GMatf d_forward_msg;
static GMatf d_backward_msg;

template <int stage>
__global__ static void fb_smooth_inplace_kernel(
	GMatf s1_ems_prob, GMatf io_forward_msg, GMatf io_backward_msg,
	float s0_ems_prob, float no_change_prob,
	int N, int w, int h) {
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int d = blockDim.z*blockIdx.z + threadIdx.z;

	if (x < w && y < h && d < N) {
		float s0, s1;
		float prev_s1;
		if (stage == FB_MSG_L2R) {
			prev_s1 = s1_ems_prob.at(0, y, d);
			for (int i = 0; i < w; i++) {
				s0 = (prev_s1 * (1.f - no_change_prob) + (1.f - prev_s1) * no_change_prob) * s0_ems_prob;
				s1 = (prev_s1 * no_change_prob + (1.f - prev_s1) * (1 - no_change_prob)) * s1_ems_prob.at(i, y, d);
				prev_s1 = s1 / (s0 + s1);
				io_forward_msg.at(i, y, d) = prev_s1;
			}
		}
		else if (stage == FB_MSG_R2L) {
			prev_s1 = s1_ems_prob.at(w - 1, y, d);
			for (int i = w - 1; i >= 0; i--) {
				s0 = prev_s1 * s1_ems_prob.at(i, y, d) * (1.f - no_change_prob) + (1.f - prev_s1) * no_change_prob * s0_ems_prob;
				s1 = prev_s1 * s1_ems_prob.at(i, y, d) * no_change_prob + (1.f - prev_s1) * (1.f - no_change_prob) * s0_ems_prob;
				prev_s1 = s1 / (s0 + s1);
				io_backward_msg.at(i, y, d) = prev_s1;
			}
		}
		else if (stage == FB_MSG_T2B) {
			prev_s1 = s1_ems_prob.at(x, 0, d);
			for (int i = 0; i < h; i++) {
				s0 = (prev_s1 * (1.f - no_change_prob) + (1.f - prev_s1) * no_change_prob) * s0_ems_prob;
				s1 = (prev_s1 * no_change_prob + (1.f - prev_s1) * (1 - no_change_prob)) * s1_ems_prob.at(x, i, d);
				prev_s1 = s1 / (s0 + s1);
				io_forward_msg.at(x, i, d) = prev_s1;
			}
		}
		else if (stage == FB_MSG_B2T) {
			prev_s1 = s1_ems_prob.at(x, h - 1, d);
			for (int i = h - 1; i >= 0; i--) {
				s0 = prev_s1 * s1_ems_prob.at(x, i, d) * (1.f - no_change_prob) + (1.f - prev_s1) * no_change_prob * s0_ems_prob;
				s1 = prev_s1 * s1_ems_prob.at(x, i, d) * no_change_prob + (1.f - prev_s1) * (1.f - no_change_prob) * s0_ems_prob;
				prev_s1 = s1 / (s0 + s1);
				io_backward_msg.at(x, i, d) = prev_s1;
			}
		}
		else if (stage == FB_POSTERIOR) {
			s0 = (1.f - io_forward_msg.at(x, y, d)) * (1.f - io_backward_msg.at(x, y, d));
			s1 = io_forward_msg.at(x, y, d) * io_backward_msg.at(x, y, d);
			s1_ems_prob.at(x, y, d) = s1 / (s0 + s1);
		}
	}
}

static int fb_smooth_batch_inplace(
	GMatf s1_ems_prob, float s0_ems_prob, float no_change_prob,
	int N, int w, int h) {

	const dim3 block_size(FB_BLOCK_WIDTH, FB_BLOCK_WIDTH, 1);
	const dim3 grid_size(DIV_CEIL(w, FB_BLOCK_WIDTH), DIV_CEIL(h, FB_BLOCK_WIDTH), N);

	const dim3 block_size_row_chain(1, FB_CHAIN_WIDTH, 1);
	const dim3 grid_size_row_chain(1, DIV_CEIL(h, FB_CHAIN_WIDTH), N);

	const dim3 block_size_col_chain(FB_CHAIN_WIDTH, 1, 1);
	const dim3 grid_size_col_chain(DIV_CEIL(w, FB_CHAIN_WIDTH), 1, N);

	d_forward_msg.create(w, h, N, true);
	d_backward_msg.create(w, h, N, true);

	fb_smooth_inplace_kernel<FB_MSG_L2R> << < grid_size_row_chain, block_size_row_chain >> >
		(s1_ems_prob, d_forward_msg, d_backward_msg, s0_ems_prob, no_change_prob, N, w, h);
	gpuErrchk;
	fb_smooth_inplace_kernel<FB_MSG_R2L> << < grid_size_row_chain, block_size_row_chain >> >
		(s1_ems_prob, d_forward_msg, d_backward_msg, s0_ems_prob, no_change_prob, N, w, h);
	gpuErrchk;
	fb_smooth_inplace_kernel<FB_POSTERIOR> << < grid_size, block_size >> >
		(s1_ems_prob, d_forward_msg, d_backward_msg, s0_ems_prob, no_change_prob, N, w, h);
	gpuErrchk;
	fb_smooth_inplace_kernel<FB_MSG_T2B> << < grid_size_col_chain, block_size_col_chain >> >
		(s1_ems_prob, d_forward_msg, d_backward_msg, s0_ems_prob, no_change_prob, N, w, h);
	gpuErrchk;
	fb_smooth_inplace_kernel<FB_MSG_B2T> << < grid_size_col_chain, block_size_col_chain >> >
		(s1_ems_prob, d_forward_msg, d_backward_msg, s0_ems_prob, no_change_prob, N, w, h);
	gpuErrchk;
	fb_smooth_inplace_kernel<FB_POSTERIOR> << < grid_size, block_size >> >
		(s1_ems_prob, d_forward_msg, d_backward_msg, s0_ems_prob, no_change_prob, N, w, h);
	gpuErrchk;

	return cudaSuccess;
}
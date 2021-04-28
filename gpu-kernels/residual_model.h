#pragma once
#include "utils.h"

// The model is evaluted on KITTI using PWC-Net with resize factor = 0.5 (flow is estimated under original scale)
// Thus, if the input flow is not resized with 0.5, an adjustment is done based on given abs_resize_factor (abs_rf).
#define EST_RF 0.5
#define FISK_A1 0.01f
#define FISK_A2 0.09f
#define FISK_B1 1.0f
#define FISK_B2 -0.0022f
#define MIN_OBS_FMAG 2.f
#define MAX_OBS_FMAG 100.f

// a/(x+b)+(1-a/b)
__device__ __inline__ static float fun_fmag_c(float fmag) {
	fmag = fminf(fmaxf(fmag*EST_RF, MIN_OBS_FMAG), MAX_OBS_FMAG);
	return FISK_B1 + FISK_B2 * fmag;
}

// a*exp(b*x)
__device__ __inline__ static float fun_fmag_scale(float fmag) {
	fmag = fminf(fmaxf(fmag*EST_RF, MIN_OBS_FMAG), MAX_OBS_FMAG);
	return FISK_A1 * expf(FISK_A2 * fmag);
}


// c*(x/s)^(-c-1)*(1+(x/s).^(-c))^(-2)/s
__device__ __inline__ static float fisk_dist_pdf(float x, float c, float scale) {
	x = fmaxf(x*EST_RF, ZDE);
	return (c * powf((x*x) / scale, -c - 1.f) * powf(1 + powf((x*x) / scale, -c), -2.f)) / scale;
}


__device__ __inline__ static float fun_rigidness(float dx1, float dy1, float dx2, float dy2, float lambda, float abs_rf) {
	const float obs_fmag = L2_NORM(dx2, dy2) / abs_rf;
	const float diff_fmag = L2_NORM(dx1 - dx2, dy1 - dy2) / abs_rf;
	const float c = fun_fmag_c(obs_fmag);
	const float s = fun_fmag_scale(obs_fmag);
	const float fisk_prob = fisk_dist_pdf(diff_fmag, c, s);
	const float mu = fisk_dist_pdf(lambda*obs_fmag, c, s);
	return fisk_prob / (fisk_prob + mu);
}


__device__ __inline__ static void fun_cost(float dx1, float dy1, float dx2, float dy2, float weight,
	float& io_cost, float& io_weight_sum, float lambda, float abs_rf) {
	io_cost -= weight * logf(fun_rigidness(dx1, dy1, dx2, dy2, lambda, abs_rf));
	io_weight_sum += weight;
}

__device__ __inline__ static float fun_depth_rigidness(float d1, float d2, float basefocal, float omega, float abs_rf) {
	const float disp1 = (basefocal / d1) / abs_rf;
	const float disp2 = (basefocal / d2) / abs_rf;
	const float obs_disp = disp2;
	const float diff_disp = fabsf(disp1 - disp2);
	const float c = fun_fmag_c(obs_disp);
	const float s = fun_fmag_scale(obs_disp);
	const float fisk_prob = fisk_dist_pdf(diff_disp, c, s);
	const float mu = fisk_dist_pdf(omega*obs_disp, c, s);
	return fisk_prob / (fisk_prob + mu);
}


__device__ __inline__ static void fun_depth_cost(float d1, float d2, float basefocal, float weight,
	float& io_cost, float& io_weight_sum, float omega, float abs_rf) {
	io_cost -= weight * logf(fun_depth_rigidness(d1, d2, basefocal, omega, abs_rf));
	io_weight_sum += weight;
}


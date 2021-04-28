#pragma once


extern int py_falign_wrapper(
	const float* depths, const float* images, const float* weights,
	float* poses_init, float* poses_ret,
	const int* connectivity,
	float* poses_covar, float* scaling_factor,
	float* visibility_mat, float* consistency_mat,
	const int N, const int w, const int h,
	const float fx, const float fy, const float cx, const float cy,
	const float vbf, const float crw,
	const bool optimize_7dof, const bool graduated_optmize,
	const int stride,
	const float consistency_residual_bound,
	const bool debug);
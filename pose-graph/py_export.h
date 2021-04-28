#pragma once

extern int py_pose_graph_optm_wrapper(
	const int* poses_idx, const float* poses,
	const int* edges_idx, const float* edges_pose, const float* edges_covar,
	float* poses_ret,
	const int n_poses, const int n_edges,
	const bool optimize_7dof,
	const bool debug);
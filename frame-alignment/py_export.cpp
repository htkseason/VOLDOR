#include "py_export.h"
#include "align_frame.h"

int py_falign_wrapper(
	const float* depths_pt, const float* images_pt, const float* weights_pt,
	float* poses_init_pt, float* poses_ret_pt,
	const int* connectivity_pt,
	float* poses_covar_pt, float* scaling_factor_pt,
	float* visibility_mat_ret_pt, float* consistency_mat_ret_pt,
	const int N, const int w, const int h,
	const float fx, const float fy, const float cx, const float cy,
	const float vbf, const float crw,
	const bool optimize_7dof, const bool graduated_optmize,
	const int stride,
	const float consistency_residual_bound,
	const bool debug) {

	vector<Mat> depths;
	vector<Mat> images;
	vector<Mat> weights;

	for (int i = 0; i < N; i++) {
		if (depths_pt)
			depths.push_back(Mat(Size(w, h), CV_32F, (void*)(depths_pt + i * w*h)));
		if (images_pt)
			images.push_back(Mat(Size(w, h), CV_32F, (void*)(images_pt + i * w*h)));
		if (weights_pt)
			weights.push_back(Mat(Size(w, h), CV_32F, (void*)(weights_pt + i * w*h)));
	}

	Mat poses;
	if (poses_init_pt)
		poses = Mat(N, 6, CV_32F, (void*)(poses_init_pt));
	poses.convertTo(poses, CV_64F);


	int pt;
	vector<pair<int, int>> connectivity; // a,b,a,b,a,b,-1
	if (connectivity_pt) {
		pt = 0;
		while (connectivity_pt[pt] != -1) {
			if (connectivity_pt[pt] < 0 || connectivity_pt[pt] >= N ||
				connectivity_pt[pt + 1] < 0 || connectivity_pt[pt + 1] >= N) {
				cout << "Invalid connectivity input" << endl;
				exit(1);
			}
			pair<int, int> con;
			con.first = connectivity_pt[pt++];
			con.second = connectivity_pt[pt++];
			connectivity.push_back(con);
		}
	}

	Mat K = (Mat_<float>(3, 3) <<
		fx, 0, cx,
		0, fy, cy,
		0, 0, 1);

	Mat poses_covar;
	Mat scaling_factor;
	Mat visibility_mat;
	Mat consistency_mat;
	
	align_frame(depths, images, weights, connectivity, 
		poses, poses_covar, scaling_factor, 
		visibility_mat, consistency_mat,
		K, vbf, crw, optimize_7dof, graduated_optmize, stride, 
		consistency_residual_bound, debug);

	poses.convertTo(poses, CV_32F);
	poses_covar.convertTo(poses_covar, CV_32F);
	scaling_factor.convertTo(scaling_factor, CV_32F);
	visibility_mat.convertTo(visibility_mat, CV_32F);
	consistency_mat.convertTo(consistency_mat, CV_32F);

	
	memcpy(poses_ret_pt, poses.data, N * 6 * sizeof(float));
	
	if (optimize_7dof)
		memcpy(poses_covar_pt, poses_covar.data, N * 7 * 7 * sizeof(float));
	else
		memcpy(poses_covar_pt, poses_covar.data, N * 6 * 6 * sizeof(float));
		
	memcpy(scaling_factor_pt, scaling_factor.data, N * sizeof(float));
	memcpy(visibility_mat_ret_pt, visibility_mat.data, N * N * sizeof(float));
	memcpy(consistency_mat_ret_pt, consistency_mat.data, N * N * sizeof(float));

	return 0;
}

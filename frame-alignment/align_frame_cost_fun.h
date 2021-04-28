#pragma once
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include "ceres/ceres.h"
#include "../gpu-kernels/gpu_kernels.h"
#include "utils.h"
#include <mutex>
#include "ceres/rotation.h"

using namespace std;
using namespace cv;


class FACostFunction
	: public ceres::CostFunction {
public:

	static mutex align_depth_eval_gpu_mutex;


	int w, h, N;

	int stride;

	int ref_fid;
	int tar_fid;


	static const int N_PARAMS_GPU = 9; //6dof pose + 1dof depth scale + 2dof color

	bool optimize_7dof;
	bool use_photo_consistency;
	bool graduated_optimize;
	bool is_biconnected;

	Mat residual_map;
	Mat jacobian_map;

	bool debug;

	double* g_ref_pose_params;
	double* g_ref_color_params;
	double* g_tar_pose_params;
	double* g_tar_color_params;

	bool eval_covar_mode = false;

	FACostFunction(
		double* _ref_pose_params,
		double* _ref_color_params,
		double* _tar_pose_params,
		double* _tar_color_params,
		int _N, int _w, int _h,
		int _fid_ref,
		int _fid_tar,
		bool _use_photo_consistency = true,
		bool _optimize_7dof = true,
		bool _graduated_optimize = true,
		bool _is_biconnected = true,
		int _stride = 4,
		bool _debug = false) :
		g_ref_pose_params(_ref_pose_params),
		g_ref_color_params(_ref_color_params),
		g_tar_pose_params(_tar_pose_params),
		g_tar_color_params(_tar_color_params),
		w(_w), h(_h), N(_N),
		ref_fid(_fid_ref),
		tar_fid(_fid_tar),
		use_photo_consistency(_use_photo_consistency),
		optimize_7dof(_optimize_7dof),
		graduated_optimize(_graduated_optimize),
		is_biconnected(_is_biconnected),
		stride(_stride),
		debug(_debug) {

		this->set_num_residuals((int)ceil((double)w / (double)stride) * (int)ceil((double)h / (double)stride));

		this->mutable_parameter_block_sizes()->push_back(6);
		if (optimize_7dof)
			this->mutable_parameter_block_sizes()->push_back(1);
		if (use_photo_consistency)
			this->mutable_parameter_block_sizes()->push_back(2);


		residual_map = Mat::zeros(Size(w, h), CV_32F);
		jacobian_map = Mat::zeros(Size(w, h), CV_MAKE_TYPE(CV_32F, N_PARAMS_GPU));

	}

	void set_eval_covar_mode(bool mode) {
		this->eval_covar_mode = mode;
	}

	void align_score(double& visibility, double& consistency, const double residual_bound = 1.0) {
		float h_ref_params_float[N_PARAMS_GPU] = { 0 };
		float h_tar_params_float[N_PARAMS_GPU] = { 0 };

		prepare_gpu_params(h_ref_params_float, h_tar_params_float,
			&g_ref_pose_params[0], &g_ref_pose_params[6], &g_ref_color_params[0],
			&g_tar_pose_params[0], &g_tar_pose_params[6], &g_tar_color_params[0]);

		// estimate unweighted residual
		FACostFunction::align_depth_eval_gpu_mutex.lock();
		align_frame_eval_gpu(ref_fid, tar_fid,
			h_ref_params_float, h_tar_params_float,
			(float*)residual_map.data, NULL,
			false);
		FACostFunction::align_depth_eval_gpu_mutex.unlock();

		int n_total = 0;
		int n_visibility = 0;
		double n_consistency = 0;
		float* it = (float*)residual_map.data;
		while (it != (float*)residual_map.dataend) {
			n_total++;
			if (isfinite(*it)) {
				n_visibility++;
				double bounded_sqr_residual = min((double)(*it)*(*it), residual_bound);
				n_consistency += (1.0 - bounded_sqr_residual / residual_bound);
			}
			it++;
		}
		visibility = (double)n_visibility / (double)n_total;
		consistency = n_consistency / (double)max(n_visibility, 1);
	}

	void prepare_gpu_params(float h_ref_params_float[N_PARAMS_GPU], float h_tar_params_float[N_PARAMS_GPU],
		const double* ref_pose_params, const double* ref_scale_params, const double* ref_color_params,
		const double* tar_pose_params, const double* tar_scale_params, const double* tar_color_params) const {
		// ==============need change if params layout changes
		h_ref_params_float[0] = (float)ref_pose_params[0];
		h_ref_params_float[1] = (float)ref_pose_params[1];
		h_ref_params_float[2] = (float)ref_pose_params[2];
		h_ref_params_float[3] = (float)ref_pose_params[3];
		h_ref_params_float[4] = (float)ref_pose_params[4];
		h_ref_params_float[5] = (float)ref_pose_params[5];
		h_ref_params_float[6] = optimize_7dof ? (float)ref_scale_params[0] : 0;
		h_ref_params_float[7] = use_photo_consistency ? (float)ref_color_params[0] : 0;
		h_ref_params_float[8] = use_photo_consistency ? (float)ref_color_params[1] : 0;

		h_tar_params_float[0] = (float)tar_pose_params[0];
		h_tar_params_float[1] = (float)tar_pose_params[1];
		h_tar_params_float[2] = (float)tar_pose_params[2];
		h_tar_params_float[3] = (float)tar_pose_params[3];
		h_tar_params_float[4] = (float)tar_pose_params[4];
		h_tar_params_float[5] = (float)tar_pose_params[5];
		h_tar_params_float[6] = optimize_7dof ? (float)tar_scale_params[0] : 0;
		h_tar_params_float[7] = use_photo_consistency ? (float)tar_color_params[0] : 0;
		h_tar_params_float[8] = use_photo_consistency ? (float)tar_color_params[1] : 0;
	}

	virtual bool Evaluate(double const* const* parameters,
		double* residuals,
		double** jacobians) const {

		int bpt;

		// ==============need change if params layout changes
		float h_ref_params_float[N_PARAMS_GPU] = { 0 };
		float h_tar_params_float[N_PARAMS_GPU] = { 0 };

		prepare_gpu_params(h_ref_params_float, h_tar_params_float,
			parameters[0], optimize_7dof ? parameters[1] : NULL, use_photo_consistency ? (optimize_7dof ? parameters[2] : parameters[1]) : NULL,
			&g_tar_pose_params[0], &g_tar_pose_params[6], &g_tar_color_params[0]);


		FACostFunction::align_depth_eval_gpu_mutex.lock();
		align_frame_eval_gpu(ref_fid, tar_fid,
			h_ref_params_float, h_tar_params_float,
			(float*)residual_map.data,
			jacobians ? (float*)jacobian_map.data : NULL,
			true);
		FACostFunction::align_depth_eval_gpu_mutex.unlock();


		if (is_biconnected && !this->eval_covar_mode) {
			jacobian_map *= 2;
		}


		for (int y = 0; y < h; y += stride) {
			for (int x = 0; x < w; x += stride) {
				const int idx = (y / stride) * (int)ceil((double)w / (double)stride) + (x / stride);

				double residual = residual_map.at<float>(y, x);

				if (!isfinite(residual)) {
					residuals[idx] = 0;
					if (jacobians) {
						bpt = 0;
						jacobians[bpt][idx * 6 + 0] = 0;
						jacobians[bpt][idx * 6 + 1] = 0;
						jacobians[bpt][idx * 6 + 2] = 0;
						jacobians[bpt][idx * 6 + 3] = 0;
						jacobians[bpt][idx * 6 + 4] = 0;
						jacobians[bpt++][idx * 6 + 5] = 0;
						if (optimize_7dof)
							jacobians[bpt++][idx] = 0;
						if (use_photo_consistency) {
							jacobians[bpt][idx * 2 + 0] = 0;
							jacobians[bpt++][idx * 2 + 1] = 0;
						}
					}
					continue;
				}


				residuals[idx] = residual;
				if (jacobians) {
					bpt = 0;
					// ==============need change if params layout changes
					jacobians[bpt][idx * 6 + 0] = jacobian_map.at<float>((y*w + x) * N_PARAMS_GPU + 0);
					jacobians[bpt][idx * 6 + 1] = jacobian_map.at<float>((y*w + x) * N_PARAMS_GPU + 1);
					jacobians[bpt][idx * 6 + 2] = jacobian_map.at<float>((y*w + x) * N_PARAMS_GPU + 2);
					jacobians[bpt][idx * 6 + 3] = jacobian_map.at<float>((y*w + x) * N_PARAMS_GPU + 3);
					jacobians[bpt][idx * 6 + 4] = jacobian_map.at<float>((y*w + x) * N_PARAMS_GPU + 4);
					jacobians[bpt++][idx * 6 + 5] = jacobian_map.at<float>((y*w + x) * N_PARAMS_GPU + 5);
					if (optimize_7dof)
						jacobians[bpt++][idx] = jacobian_map.at<float>((y*w + x) * N_PARAMS_GPU + 6);
					if (use_photo_consistency) {
						jacobians[bpt][idx * 2 + 0] = jacobian_map.at<float>((y*w + x) * N_PARAMS_GPU + 7);
						jacobians[bpt++][idx * 2 + 1] = jacobian_map.at<float>((y*w + x) * N_PARAMS_GPU + 8);
					}
					// ==============need change if params layout changes
				}
			}
		}

		if (debug) {
			imshow("residual map-" + to_string(ref_fid) + "-" + to_string(tar_fid), residual_map.mul(residual_map));
			//imshow("residual map-" + to_string(ref_fid) + "-" + to_string(tar_fid), residual_map);

			if (waitKey(1) == 'q')
				exit(1);
		}

		return true;
	}
};

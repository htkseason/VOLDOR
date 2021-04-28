#pragma once
#include "utils.h"

struct Config {
	//vector<string> flow_names;

	// depth prior related
	float omega = 0.15f; //depth prior rigidness strictness
	float disp_delta = 1.f; //disparity depth prior weight
	float delta = 0.5f; //depth prior weight
	float basefocal = 0; //baseline x focal

	// robust gaussian fit related
	int rg_refine = true;
	int rg_refine_last_only = true;
	float rg_trunc_sigma = 3.f;
	float rg_covar_reg_lambda = 0.001f; //lediot_wolf regulization for covariance matrix
	float rg_pose_scaling = 100.f; //pose scaling
	int rg_max_iters = 100;
	float rg_epsilon = 1e-5f;

	// input-params
	float resize_factor = 1.0f; //(deprecated, now resize is done in slam logic)
	float abs_resize_factor = 1.0f; //resize factor related to the size that optical flow is estimated from. (useful to residual model)
	float fx = 0.0f, fy = 0.0f;
	float cx = 0.0f, cy = 0.0f;
	int exclusive_gpu_context = true; //if only one voldor instance is running under the gpu context, set this to true, a lot of optimization will be applied

	// debug related
	bool debug = false; //debug mode
	bool silent = false; //silent mode
	bool save_everything = false; //(deprecated)
	int viz_img_per_row = 2; //debug visualizer image per row
	float viz_depth_scale = 5; //inv-depth visualization scaling

	// hyper-params
	float lambda = 0.15f; //rigidness strictness
	float meanshift_kernel_var = 0.1f; //meanshift kernel variance 
	float meanshift_rvec_scale = 25.0f; //rotation vector scaling before meanshift
	int norm_world_scale = true; //normalize mean translation norm to 1.0, improve the robustness for monocular initialization

	// pose sampling related
	int cpu_p3p = false; //do p3p on cpu
	int lambdatwist = true; //use lambdatwist instead of ap3p
	int n_poses_to_sample = 8192;
	float pose_sample_min_depth = 0.1f;
	float pose_sample_max_depth = 1000.0f;
	int max_trace_on_flow = 3; //maximum times of tracking only with optical flow
	float rigidness_threshold = 0.5f;
	float rigidness_sum_threshold = 1.f;

	// truncation related
	float trunc_rigidness_density = 0.05f;
	float trunc_sample_density = 0.001f;
	float no_trunc_iters = 2;
	int max_iters = 5;
	int min_iters_after_trunc = 3;

	// fb smooth related
	int fb_smooth = true; //use forward-backward smoothing on rigidness maps
	float fb_emm = 0.5f; //emmission probablity
	float fb_no_change_prob = 0.9f; //no change probability

	// depth update related
	int optimize_depth = true;
	int depth_rand_samples = 10;
	int depth_global_prop_step = 8;
	int depth_local_prop_width = 32;
	float depth_range_factor = 1.f;

	// meanshift related
	int meanshift_max_iters = 100;
	int meanshift_max_init_trials = 20;
	float meanshift_good_init_confidence = 0.5f;
	float meanshift_epsilon = 1e-5;

	// KITTI ground-height-estimation 
	// NOTE this function is not used anymore, the code if left here for reference to the paper
	int kitti_estimate_ground = false;
	int kitti_ground_holo_width = 5;
	float kitti_ground_roi = 0.4f;
	float kitti_ground_meanshift_kernel_var = 0.01f;

	template <typename T>
	static void str_to_arg(string str, T& arg) {
		switch (*(typeid(arg).name()))
		{
		case 'i':
			arg = stoi(str);
		case 'l':
			arg = stol(str);
		case 'f':
			arg = stof(str);
		case 'd':
			arg = stod(str);
		default:
			break;
		}
	}

	template <typename T>
	static T safe_arr_access(vector<T> arr, size_t i) {
		if (i>=0 && i<arr.size())
			return arr[i];
		else
			cout << "Config array index out of bound." << endl;
			exit(1);
	}

	void read_config(vector<string> cfg_strs) {

		for (int i = 0; i < cfg_strs.size(); i++) {
			if (0) {
			}

			else if (cfg_strs[i] == "--basefocal")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->basefocal);
			else if (cfg_strs[i] == "--omega")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->omega);
			else if (cfg_strs[i] == "--disp_delta")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->disp_delta);
			else if (cfg_strs[i] == "--delta")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->delta);

			else if (cfg_strs[i] == "--rg_refine")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->rg_refine);
			else if (cfg_strs[i] == "--rg_refine_last_only")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->rg_refine_last_only);
			else if (cfg_strs[i] == "--rg_trunc_sigma")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->rg_trunc_sigma);
			else if (cfg_strs[i] == "--rg_covar_reg_lambda")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->rg_covar_reg_lambda);
			else if (cfg_strs[i] == "--rg_epsilon")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->rg_epsilon);
			else if (cfg_strs[i] == "--rg_max_iters")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->rg_max_iters);
			else if (cfg_strs[i] == "--rg_pose_scaling")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->rg_pose_scaling);


			else if (cfg_strs[i] == "--resize_factor")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->resize_factor);
			else if (cfg_strs[i] == "--abs_resize_factor")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->abs_resize_factor);
			else if (cfg_strs[i] == "--fx")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->fx);
			else if (cfg_strs[i] == "--fy")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->fy);
			else if (cfg_strs[i] == "--cx")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->cx);
			else if (cfg_strs[i] == "--cy")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->cy);


			else if (cfg_strs[i] == "--debug")
				this->debug = true;
			else if (cfg_strs[i] == "--silent")
				this->silent = true;
			else if (cfg_strs[i] == "--save_everything")
				this->save_everything = true;
			else if (cfg_strs[i] == "--viz_img_per_row")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->viz_img_per_row);
			else if (cfg_strs[i] == "--viz_depth_scale")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->viz_depth_scale);
			else if (cfg_strs[i] == "--exclusive_gpu_context")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->exclusive_gpu_context);


			else if (cfg_strs[i] == "--lambda")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->lambda);
			else if (cfg_strs[i] == "--meanshift_kernel_var")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->meanshift_kernel_var);
			else if (cfg_strs[i] == "--meanshift_rvec_scale")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->meanshift_rvec_scale);
			else if (cfg_strs[i] == "--norm_world_scale")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->norm_world_scale);

			else if (cfg_strs[i] == "--cpu_p3p")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->cpu_p3p);
			else if (cfg_strs[i] == "--lambdatwist")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->lambdatwist);
			else if (cfg_strs[i] == "--max_trace_on_flow")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->max_trace_on_flow);
			else if (cfg_strs[i] == "--n_poses_to_sample")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->n_poses_to_sample);
			else if (cfg_strs[i] == "--pose_sample_min_depth")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->pose_sample_min_depth);
			else if (cfg_strs[i] == "--pose_sample_max_depth")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->pose_sample_max_depth);
			else if (cfg_strs[i] == "--rigidness_threshold")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->rigidness_threshold);
			else if (cfg_strs[i] == "--rigidness_sum_threshold")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->rigidness_sum_threshold);


			else if (cfg_strs[i] == "--trunc_rigidness_density")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->trunc_rigidness_density);
			else if (cfg_strs[i] == "--trunc_sample_density")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->trunc_sample_density);
			else if (cfg_strs[i] == "--max_iters")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->max_iters);
			else if (cfg_strs[i] == "--no_trunc_iters")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->no_trunc_iters);
			else if (cfg_strs[i] == "--min_iters_after_trunc")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->min_iters_after_trunc);

			else if (cfg_strs[i] == "--fb_smooth")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->fb_smooth);
			else if (cfg_strs[i] == "--fb_emm")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->fb_emm);
			else if (cfg_strs[i] == "--fb_no_change_prob")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->fb_no_change_prob);

			else if (cfg_strs[i] == "--optimize_depth")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->optimize_depth);
			else if (cfg_strs[i] == "--depth_rand_samples")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->depth_rand_samples);
			else if (cfg_strs[i] == "--depth_global_prop_step")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->depth_global_prop_step);
			else if (cfg_strs[i] == "--depth_local_prop_width")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->depth_local_prop_width);
			else if (cfg_strs[i] == "--depth_range_factor")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->depth_range_factor);


			else if (cfg_strs[i] == "--meanshift_max_iters")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->meanshift_max_iters);
			else if (cfg_strs[i] == "--meanshift_max_init_trials")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->meanshift_max_init_trials);
			else if (cfg_strs[i] == "--meanshift_good_init_confidence")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->meanshift_good_init_confidence);
			else if (cfg_strs[i] == "--meanshift_epsilon")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->meanshift_epsilon);


			else if (cfg_strs[i] == "--kitti_estimate_ground")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->kitti_estimate_ground);
			else if (cfg_strs[i] == "--kitti_ground_holo_width")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->kitti_ground_holo_width);
			else if (cfg_strs[i] == "--kitti_ground_roi")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->kitti_ground_roi);
			else if (cfg_strs[i] == "--kitti_ground_meanshift_kernel_var")
				str_to_arg(safe_arr_access(cfg_strs, ++i), this->kitti_ground_meanshift_kernel_var);

			else {
				cout << "Invalid input config : " << cfg_strs[i] <<endl;
				exit(1);
			}



		}
	}

	void print_info() {
		cout << endl << "================= Configurations =================" << endl;
		cout << "omega = " << omega << endl;
		cout << "delta = " << delta << endl;
		cout << "disp_delta = " << disp_delta << endl;
		cout << "basefocal = " << basefocal << endl;

		cout << "rg_refine = " << rg_refine << endl;
		cout << "rg_refine_last_only = " << rg_refine_last_only << endl;
		cout << "rg_trunc_sigma = " << rg_trunc_sigma << endl;
		cout << "rg_covar_reg_lambda = " << rg_covar_reg_lambda << endl;
		cout << "rg_pose_scaling = " << rg_pose_scaling << endl;
		cout << "rg_epsilon = " << rg_epsilon << endl;
		cout << "rg_max_iters = " << rg_max_iters << endl;

		cout << "resize_factor = " << resize_factor << endl;
		cout << "abs_resize_factor = " << abs_resize_factor << endl;
		cout << "fx = " << fx << endl;
		cout << "fy = " << fy << endl;
		cout << "cx = " << cx << endl;
		cout << "cy = " << cy << endl;

		cout << "debug = " << debug << endl;
		cout << "silent = " << silent << endl;
		cout << "viz_img_per_row = " << viz_img_per_row << endl;
		cout << "viz_depth_scale = " << viz_depth_scale << endl;

		cout << "lambda = " << lambda << endl;
		cout << "meanshift_kernel_var = " << meanshift_kernel_var << endl;
		cout << "meanshift_rvec_scale = " << meanshift_rvec_scale << endl;

		cout << "cpu_p3p = " << cpu_p3p << endl;
		cout << "lambdatwist = " << lambdatwist << endl;
		cout << "max_trace_on_flow = " << max_trace_on_flow << endl;
		cout << "n_poses_to_sample = " << n_poses_to_sample << endl;
		cout << "pose_sample_min_depth = " << pose_sample_min_depth << endl;
		cout << "pose_sample_max_depth = " << pose_sample_max_depth << endl;
		cout << "rigidness_threshold = " << rigidness_threshold << endl;
		cout << "rigidness_sum_threshold = " << rigidness_sum_threshold << endl;

		cout << "trunc_rigidness_density = " << trunc_rigidness_density << endl;
		cout << "trunc_sample_density = " << trunc_sample_density << endl;
		cout << "no_trunc_iters = " << no_trunc_iters << endl;
		cout << "max_iters = " << max_iters << endl;
		cout << "min_iters_after_trunc = " << min_iters_after_trunc << endl;


		cout << "fb_smooth = " << fb_smooth << endl;
		cout << "fb_emm = " << fb_emm << endl;
		cout << "fb_no_change_prob = " << fb_no_change_prob << endl;

		cout << "depth_rand_samples = " << depth_rand_samples << endl;
		cout << "depth_global_prop_step = " << depth_global_prop_step << endl;
		cout << "depth_local_prop_width = " << depth_local_prop_width << endl;
		cout << "depth_range_factor = " << depth_range_factor << endl;

		cout << "meanshift_max_iters = " << meanshift_max_iters << endl;
		cout << "meanshift_init_trials = " << meanshift_max_init_trials << endl;
		cout << "meanshift_good_init_confidence = " << meanshift_good_init_confidence << endl;
		cout << "meanshift_epsilon = " << meanshift_epsilon << endl;


		cout << "kitti_estimate_ground = " << kitti_estimate_ground << endl;
		cout << "kitti_ground_holo_width = " << kitti_ground_holo_width << endl;
		cout << "kitti_ground_roi = " << kitti_ground_roi << endl;
		cout << "kitti_ground_meanshift_kernel_var = " << kitti_ground_meanshift_kernel_var << endl;


		cout << "==================================================" << endl << endl;
	}
};
#include "align_frame.h"

mutex FACostFunction::align_depth_eval_gpu_mutex;

void align_frame(
	vector<Mat> depths,
	vector<Mat> images,
	vector<Mat> weights,
	vector<pair<int, int>> connectivity,
	Mat& poses,
	Mat& poses_covar,
	Mat& scaling_factor,
	Mat& visibility_mat,
	Mat& consistency_mat,
	Mat K,
	float vbf,
	float crw,
	bool optimize_7dof,
	bool graduated_optimize,
	int stride,
	float consistency_residual_bound,
	bool debug) {


	//const int N_POSE_PARAMS = optimize_7dof ? 7 : 6; // 6dof group pose + 1dof scale (optional)
	//const int N_COLOR_PARAMS = 2; // 2dof color adjust

	const bool use_photo_consistency = images.size() > 0 && crw > 0;

	int w = depths[0].cols;
	int h = depths[0].rows;
	int N = depths.size();

	if (weights.empty()) {
		for (int i = 0; i < N; i++)
			weights.push_back(Mat::ones(Size(w, h), CV_32F));
	}


	if (connectivity.empty()) {
		for (int ref_fid = 0; ref_fid < N; ref_fid++) {
			for (int tar_fid = 0; tar_fid < N; tar_fid++) {
				if (ref_fid == tar_fid)
					continue;
				connectivity.push_back(make_pair(ref_fid, tar_fid));
			}
		}
	}

	Mat pose_params = Mat::zeros(N, optimize_7dof ? 7 : 6, CV_64F);
	Mat color_params = Mat::zeros(N, 2, CV_64F);

	if (!poses.empty()) {
		if (poses.depth() != CV_64F)
			poses.convertTo(poses, CV_64F);
		if (poses.rows != N || poses.cols != 6) {
			cout << "Invalid given pose size" << endl;
			exit(1);
		}
		for (int i = 0; i < N; i++) {
			memcpy(&pose_params.at<double>(i, 0), &poses.at<double>(i, 0), 6 * sizeof(double));
		}
	}


	
	if (debug) {
		for (int i = 0; i < N; i++) {
			imshow("depth-" + to_string(i), 8 / depths[i]);
			imshow("weight-" + to_string(i), weights[i]);
			if (use_photo_consistency)
				imshow("image-" + to_string(i), images[i]);
		}
		waitKey(0);
	}



	float** h_depths = get_mat_arr_pointer<float>(depths);
	float** h_weights = get_mat_arr_pointer<float>(weights);

	if (use_photo_consistency) {
		float** h_images = get_mat_arr_pointer<float>(images);
		align_frame_init_gpu(h_images, h_depths, h_weights, (float*)K.data, vbf, crw, N, w, h);
		delete[] h_images;
	}
	else {
		align_frame_init_gpu(NULL, h_depths, h_weights, (float*)K.data, vbf, crw, N, w, h);
	}

	delete[] h_depths;


	ceres::Problem problem;

	map<pair<int, int>, ceres::CostFunction*> cost_funs;

	for (int i = 0; i < connectivity.size(); i++) {
		int ref_fid = connectivity[i].first;
		int tar_fid = connectivity[i].second;

		bool is_biconnected = find(connectivity.begin(), connectivity.end(), pair<int, int>(tar_fid, ref_fid)) != connectivity.end();

		double* _ref_pose_params = &pose_params.at<double>(ref_fid, 0);
		double* _ref_color_params = &color_params.at<double>(ref_fid, 0);
		double* _tar_pose_params = &pose_params.at<double>(tar_fid, 0);
		double* _tar_color_params = &color_params.at<double>(tar_fid, 0);


		ceres::CostFunction* cost_function = new FACostFunction(
			_ref_pose_params, _ref_color_params,
			_tar_pose_params, _tar_color_params,
			N, w, h, ref_fid, tar_fid,
			use_photo_consistency, optimize_7dof, graduated_optimize, is_biconnected, stride, debug);
		if (optimize_7dof) {
			if (use_photo_consistency) {
				problem.AddResidualBlock(cost_function, NULL,
					_ref_pose_params, _ref_pose_params + 6, _ref_color_params);
			}
			else {
				problem.AddResidualBlock(cost_function, NULL,
					_ref_pose_params, _ref_pose_params + 6);
			}
		}
		else {
			if (use_photo_consistency) {
				problem.AddResidualBlock(cost_function, NULL,
					_ref_pose_params, _ref_color_params);
			}
			else {
				problem.AddResidualBlock(cost_function, NULL,
					_ref_pose_params);
			}
		}

		cost_funs[connectivity[i]] = cost_function;
	}

	//problem.SetParameterBlockConstant(&pose_params.at<double>(0, 0));
	//problem.SetParameterBlockConstant(&color_params.at<double>(0, 0));

	ceres::Solver::Options options;
	options.update_state_every_iteration = true; //important!
	options.max_num_iterations = 100;
	options.minimizer_type = ceres::TRUST_REGION;
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	//options.trust_region_strategy_type = ceres::DOGLEG;
	options.initial_trust_region_radius = 1;
	if (debug) {
		options.num_threads = 1;
		options.minimizer_progress_to_stdout = true;
	}
	else {
		options.num_threads = min(problem.NumResidualBlocks(), 8);
		options.minimizer_progress_to_stdout = false;
	}

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);


	// evaluate covariance matrix
	ceres::Covariance::Options options_covar;
	options_covar.num_threads = N;

	auto it = cost_funs.begin();
	while (it != cost_funs.end()) {
		pair<int, int> pair = it->first;
		FACostFunction* cost_fun = (FACostFunction*)(it->second);
		cost_fun->set_eval_covar_mode(true);
		it++;
	}

	vector<pair<const double*, const double*> > pose_params_covar;
	for (int i = 0; i < N; i++) {
		pose_params_covar.push_back(make_pair(&pose_params.at<double>(i, 0), &pose_params.at<double>(i, 0)));
		if (optimize_7dof)
			pose_params_covar.push_back(make_pair(&pose_params.at<double>(i, 6), &pose_params.at<double>(i, 6)));
	}

	poses_covar = Mat::zeros(N * 6, 6, CV_64F);
	ceres::Covariance covariance(options_covar);
	if (covariance.Compute(pose_params_covar, &problem)) {
		for (int i = 0; i < N; i++) {
			covariance.GetCovarianceBlock(&pose_params.at<double>(i, 0), &pose_params.at<double>(i, 0), &poses_covar.at<double>(i * 6, 0));
		}
		if (optimize_7dof) {
			Mat pose6_covar = poses_covar;
			poses_covar = Mat::zeros(N * 7, 7, CV_64F);
			for (int i = 0; i < N; i++) {
				pose6_covar(Range(i * 6, i * 6 + 6), Range(0, 6)).copyTo(poses_covar(Range(i * 7, i * 7 + 6), Range(0, 6)));
				covariance.GetCovarianceBlock(&pose_params.at<double>(i, 6), &pose_params.at<double>(i, 6), &poses_covar.at<double>(i * 7 + 6, 6));
			}
		}
	}



	// return frame poses
	poses = Mat::zeros(N, 6, CV_64F);
	for (int i = 0; i < N; i++) {
		memcpy(&poses.at<double>(i, 0), &pose_params.at<double>(i, 0), 6 * sizeof(double));
	}


	// return scaling factor
	scaling_factor = Mat::ones(N, 1, CV_64F);
	if (optimize_7dof) {
		for (int i = 0; i < N; i++) {
			scaling_factor.at<double>(i) = exp(pose_params.at<double>(i, 6));
		}
	}

	// compute score
	visibility_mat = Mat::zeros(N, N, CV_64F);
	consistency_mat = Mat::zeros(N, N, CV_64F);
	double dbl_nan = std::nan("");
	for (int i1 = 0; i1 < N; i1++) {
		for (int i2 = 0; i2 < N; i2++) {
			pair<int, int> con(i1, i2);
			if (find(connectivity.begin(), connectivity.end(), con) != connectivity.end()) {
				double visibility = 0, consistency = 0;
				((FACostFunction*)(cost_funs[con]))->align_score(visibility, consistency, consistency_residual_bound);
				visibility_mat.at<double>(i1, i2) = visibility;
				consistency_mat.at<double>(i1, i2) = consistency;
			}
			else {
				visibility_mat.at<double>(i1, i2) = dbl_nan;
				consistency_mat.at<double>(i1, i2) = dbl_nan;
			}
		}
	}


	if (debug) {
		cout << summary.FullReport() << endl;
		waitKey(0);
		destroyAllWindows();
	}
	else {
		//cout << summary.FullReport() << endl;
	}

}


#include "voldor.h"


void VOLDOR::init(vector<Mat> _flows,
	Mat _disparity,
	Mat _disparity_pconf,
	vector<Mat> _depth_priors,
	vector<Vec6f> _depth_prior_poses,
	vector<Mat> _depth_prior_pconfs) {

	// clear
	flows.clear();
	rigidnesses.clear();
	cams.clear();
	iters_cur = 0;
	iters_remain = cfg.max_iters;

	// we assume flow, disparity and camera parameters need apply resize
	// while depth_prior and their poses are pre-resized since they are usually previous VOLDOR result

	// copy flows
	for (int i = 0; i < _flows.size(); i++) {
		Mat flow = _flows[i].clone();
		if (cfg.resize_factor != 1) {
			cv::resize(flow, flow, Size(0, 0), cfg.resize_factor, cfg.resize_factor);
			flow *= cfg.resize_factor;
		}
		flows.push_back(flow);
	}

	// convert disparity to general depth prior
	if (!_disparity.empty()) {
		Mat depth_prior = cfg.basefocal / _disparity;
		if (cfg.resize_factor != 1) {
			cv::resize(depth_prior, depth_prior, Size(0, 0), cfg.resize_factor, cfg.resize_factor);
			depth_prior *= cfg.resize_factor;
		}
		depth_priors.push_back(depth_prior);

		if (!_disparity_pconf.empty())
			depth_prior_pconfs.push_back(_disparity_pconf);
		else
			depth_prior_pconfs.push_back(Mat::ones(_disparity.rows, _disparity.cols, CV_32F));

		Camera cam;
		cam.R = Mat::eye(3, 3, CV_32F);
		cam.t = Mat::zeros(3, 1, CV_32F);
		depth_prior_poses.push_back(cam);
	}

	// copy depth priors
	for (int i = 0; i < _depth_priors.size(); i++) {
		depth_priors.push_back(_depth_priors[i].clone());

		if (_depth_prior_pconfs.size() > 0)
			depth_prior_pconfs.push_back(_depth_prior_pconfs[i].clone());
		else
			depth_prior_pconfs.push_back(Mat::ones(_depth_priors[i].rows, _depth_priors[i].cols, CV_32F));

		Camera cam;
		Vec3f rvec = Vec3f(_depth_prior_poses[i][0], _depth_prior_poses[i][1], _depth_prior_poses[i][2]);
		Vec3f tvec = Vec3f(_depth_prior_poses[i][3], _depth_prior_poses[i][4], _depth_prior_poses[i][5]);
		Rodrigues(rvec, cam.R);
		cam.t.at<Vec3f>(0) = tvec;
		depth_prior_poses.push_back(cam);
	}

	// apply resize to config params to make life easier
	if (cfg.resize_factor != 1) {
		cfg.fx *= cfg.resize_factor;
		cfg.fy *= cfg.resize_factor;
		cfg.cx *= cfg.resize_factor;
		cfg.cy *= cfg.resize_factor;
		// if want to rescale world size, apply resize to basefocal
		//cfg.basefocal *= cfg.resize_factor;
	}


	// init params
	w = flows[0].cols;
	h = flows[0].rows;
	n_flows = flows.size();
	n_flows_init = flows.size();
	n_depth_priors = depth_priors.size();

	has_disparity = !_disparity.empty(); // tell gpu depth priors do not have disparity prior, so it does not apply disp_delta

	// init dp confs
	for (int i = 0; i < n_depth_priors; i++)
		depth_prior_confs.push_back(Mat::ones(Size(w, h), CV_32F));

	// init rigidnesses
	for (int i = 0; i < n_flows; i++)
		rigidnesses.push_back(Mat::ones(Size(w, h), CV_32F));


	// init cams
	Mat K = (Mat_<float>(3, 3) <<
		cfg.fx, 0, cfg.cx,
		0, cfg.fy, cfg.cy,
		0, 0, 1);
	Mat K_inv = K.inv();
	for (int i = 0; i < n_flows; i++) {
		Camera cam;
		cam.K = K.clone();
		cam.K_inv = K_inv.clone();
		cams.push_back(cam);
	}

	// init depth
	if (n_depth_priors > 0) {
		// if disparity is present, depth map is initialized with that
		// otherwise, optimize_depth will fuse priors
		depth = depth_priors[0].clone();
		if (_disparity.empty())
			optimize_depth(OD_ONLY_USE_DEPTH_PRIOR); 
	}
	else {
		depth = Mat::ones(Size(w, h), CV_32F);
	}

	if (!cfg.silent) {
		cout << n_flows_init << " flows loaded" << endl;
		cout << n_depth_priors << " depth priors loaded" << endl;
		cout << "w = " << w << ", h = " << h << endl << endl;
		cout << "============================================" << endl;
	}
}

int VOLDOR::solve() {
	if (cfg.debug)
		debug();
	if (n_depth_priors == 0)
		bootstrap();
	while (iters_remain > 0 && n_flows > 0) {
		iters_cur++;
		iters_remain--;
		optimize_cameras();
		optimize_depth(cfg.optimize_depth ? OD_DEFAULT : OD_UPDATE_RIGIDNESS_ONLY);
		if (cfg.norm_world_scale && n_depth_priors == 0)
			normalize_world_scale();
		if (cfg.debug)
			debug();
	}
	destroyAllWindows();
	if (cfg.kitti_estimate_ground)
		estimate_kitti_ground();
	return iters_cur;
}

void VOLDOR::bootstrap() {
	tic();

	estimate_camera_pose_epipolar(flows[0], cams[0]);
	estimate_depth_closed_form(flows[0], depth, cams[0]);

	if (!cfg.silent) {
		cams[0].print_info();
		toc("bootstrap");
		cout << "============================================" << endl;
	}
}

void VOLDOR::optimize_cameras() {
	tic();

	bool allow_trunc = iters_cur > cfg.no_trunc_iters;

	// optimize camera pose
	for (int i = 0; i < n_flows; i++) {
		cams[i].pose_rigidness_density = (float)sum(rigidnesses[i])[0] / (float)(w*h);

		int optimize_success = 0;
		if (!allow_trunc || cams[i].pose_rigidness_density > cfg.trunc_rigidness_density) {
			optimize_success = optimize_camera_pose(flows, rigidnesses, depth, cams,
				n_flows, i,
				cams[i].pose_sample_count == 0 ? false : true, //successive pose?
				cfg.rg_refine && (!cfg.rg_refine_last_only || iters_remain == 0), //rg_refine?
				!cfg.exclusive_gpu_context || (iters_cur == 1 && i == 0),  //update batch instance?
				i == 0, //update iter instance?
				cfg);
		}

		if (!cfg.silent)
			cams[i].print_info();

		if (!optimize_success || // check failure
			(allow_trunc && cams[i].pose_density < cfg.trunc_sample_density)) { // truncate confidence
			if (!cfg.silent)
				cout << "truncated at camera " << i << endl;
			iters_remain = max(iters_remain, cfg.min_iters_after_trunc);
			n_flows = i;
			break;
		}
	}

	if (!cfg.silent) {
		toc("optimize_cameras");
		cout << "============================================" << endl;
	}
}

void VOLDOR::optimize_depth(OPTIMIZE_DEPTH_FLAG flag) {
	if (n_flows == 0 && n_depth_priors == 0)
		return;
	tic();

	// optimize depth
	float** h_flows = NULL;
	float** h_rigidnesses = NULL;
	float** h_Rs = NULL;
	float** h_ts = NULL;

	if (n_flows > 0 && flag != OD_ONLY_USE_DEPTH_PRIOR) {
		h_flows = new float*[n_flows];
		h_rigidnesses = new float*[n_flows];
		h_Rs = new float*[n_flows];
		h_ts = new float*[n_flows];

		for (int i = 0; i < n_flows; i++) {
			h_flows[i] = (float*)flows[i].data;
			h_rigidnesses[i] = (float*)rigidnesses[i].data;
			h_Rs[i] = (float*)cams[i].R.data;
			h_ts[i] = (float*)cams[i].t.data;
		}
	}


	float** h_depth_priors = NULL;
	float** h_depth_prior_pconfs = NULL;
	float** h_depth_prior_confs = NULL;
	float** h_dp_Rs = NULL;
	float** h_dp_ts = NULL;

	if (n_depth_priors > 0) {
		h_depth_priors = new float*[n_depth_priors];
		h_depth_prior_pconfs = new float*[n_depth_priors];
		h_depth_prior_confs = new float*[n_depth_priors];
		h_dp_Rs = new float*[n_depth_priors];
		h_dp_ts = new float*[n_depth_priors];

		for (int i = 0; i < n_depth_priors; i++) {
			h_depth_priors[i] = (float*)depth_priors[i].data;
			h_depth_prior_pconfs[i] = (float*)depth_prior_pconfs[i].data;
			h_depth_prior_confs[i] = (float*)depth_prior_confs[i].data;
			h_dp_Rs[i] = (float*)depth_prior_poses[i].R.data;
			h_dp_ts[i] = (float*)depth_prior_poses[i].t.data;
		}
	}
	if (!cfg.exclusive_gpu_context || iters_cur == 0 || iters_cur == 1) {
		// gpu cache need update
		// iters_cur==0 is the call for fusing depth priors
		// iters_cur==1 is the first call for fusing everything
		optimize_depth_gpu(
			h_flows,
			h_rigidnesses, h_rigidnesses,
			h_depth_priors, h_depth_prior_pconfs,
			h_depth_prior_confs, h_depth_prior_confs,
			(float*)depth.data, (float*)depth.data,
			(float*)cams[0].K.data,
			h_Rs, h_ts,
			h_dp_Rs, h_dp_ts,
			cfg.abs_resize_factor,
			(flag == OD_ONLY_USE_DEPTH_PRIOR) ? 0 : n_flows, n_depth_priors, w, h, cfg.basefocal,
			cfg.depth_rand_samples, cfg.depth_global_prop_step, cfg.depth_local_prop_width,
			cfg.lambda, cfg.omega, has_disparity ? cfg.disp_delta : -1, cfg.delta,
			cfg.fb_smooth, cfg.fb_emm, cfg.fb_no_change_prob,
			cfg.depth_range_factor,
			flag == OD_UPDATE_RIGIDNESS_ONLY);
	}
	else {
		// update minimal caches
		// some inputs will need add back if latter changed outside GPU
		optimize_depth_gpu(
			NULL,
			NULL, h_rigidnesses,
			NULL, NULL,
			NULL, h_depth_prior_confs,
			NULL, (float*)depth.data,
			NULL,
			h_Rs, h_ts,
			NULL, NULL,
			cfg.abs_resize_factor,
			n_flows, n_depth_priors, w, h, cfg.basefocal,
			cfg.depth_rand_samples, cfg.depth_global_prop_step, cfg.depth_local_prop_width,
			cfg.lambda, cfg.omega, has_disparity ? cfg.disp_delta : -1, cfg.delta,
			cfg.fb_smooth, cfg.fb_emm, cfg.fb_no_change_prob,
			cfg.depth_range_factor,
			flag == OD_UPDATE_RIGIDNESS_ONLY);
	}


	delete[] h_flows;
	delete[] h_rigidnesses;
	delete[] h_Rs;
	delete[] h_ts;

	delete[] h_depth_priors;
	delete[] h_depth_prior_confs;
	delete[] h_dp_Rs;
	delete[] h_dp_ts;

	if (!cfg.silent) {
		toc("optimize_depth");
		cout << "============================================" << endl;
	}
}

void VOLDOR::normalize_world_scale() {
	// normalize world scale 
	float world_scale = 0;
	for (int i = 0; i < n_flows; i++)
		world_scale += norm(cams[i].t);
	for (int i = 0; i < n_flows; i++)
		cams[i].t *= (n_flows / world_scale);
	depth *= (n_flows / world_scale);
}


void VOLDOR::estimate_kitti_ground() {
	tic();

	Rect roi = Rect(w*0.5*(1 - cfg.kitti_ground_roi), h*(1 - cfg.kitti_ground_roi), w*cfg.kitti_ground_roi, h*cfg.kitti_ground_roi);
	ground = estimate_kitti_ground_plane(depth, roi, cams[0].K, cfg.kitti_ground_holo_width, cfg.kitti_ground_meanshift_kernel_var);

	if (!cfg.silent) {
		ground.print_info();
		toc("estimate_ground");
		cout << "============================================" << endl;
	}
}

void VOLDOR::save_result(string save_dir) {
	FILE* fs;

	// save depth
	imwrite(save_dir + PATH_SEPARATOR + "depth.png", depth);

	// save camera poses
	fs = NULL;
	fs = fopen((save_dir + PATH_SEPARATOR + "camera_pose.txt").c_str(), "w");
	assert(fs != NULL);
	for (int i = 0; i < n_flows; i++)
		cams[i].save(fs);
	fclose(fs);

	// save normalized rigidness sum
	Mat rigidness_sum = Mat::zeros(Size(w, h), CV_32F);
	for (int i = 0; i < n_flows; i++)
		rigidness_sum += rigidnesses[i];
	imwrite(save_dir + PATH_SEPARATOR + "rigidness_sum.png", 255 * rigidness_sum / (float)n_flows);

	if (cfg.kitti_estimate_ground) {
		// save ground info
		fs = NULL;
		fs = fopen((save_dir + PATH_SEPARATOR + "height.txt").c_str(), "w");
		assert(fs != NULL);
		ground.save(fs);
		fclose(fs);
	}

	if (cfg.save_everything) {
		// save rigidness maps and flow viz
		for (int i = 0; i < flows.size(); i++)
			imwrite(save_dir + PATH_SEPARATOR + "flow-" + to_string(i) + ".png", 255 * vis_flow(flows[i]));
		for (int i = 0; i < rigidnesses.size(); i++)
			imwrite(save_dir + PATH_SEPARATOR + "rigidness-" + to_string(i) + ".png", rigidnesses[i]);
		for (int i = 0; i < depth_prior_confs.size(); i++)
			imwrite(save_dir + PATH_SEPARATOR + "depth_prior_conf-" + to_string(i) + ".png", depth_prior_confs[i]);
	}

	if (!cfg.silent) {
		cout << "results saved to " << save_dir << endl;
		cout << "============================================" << endl;
	}
}

void VOLDOR::debug() {
	if (cfg.kitti_estimate_ground)
		estimate_kitti_ground();

	int vis_img_per_col = div_ceil(n_flows_init, cfg.viz_img_per_row);
	Mat rigidnesses_world = Mat::zeros(Size(w * cfg.viz_img_per_row, h*vis_img_per_col), CV_32F);
	for (int i = 0; i < n_flows_init; i++)
		rigidnesses[i].copyTo(rigidnesses_world(Rect(w*(i / vis_img_per_col), h*(i % vis_img_per_col), w, h)));

	imshow("rigidnesses_world", rigidnesses_world);
	imshow("depth_est", cfg.viz_depth_scale / depth);
	for (int i = 0; i < n_depth_priors; i++) {
		imshow("depth_prior_" + to_string(i), cfg.viz_depth_scale / depth_priors[i]);
		imshow("depth_prior_conf_" + to_string(i), depth_prior_confs[i]);
		imshow("depth_prior_pconf_" + to_string(i), depth_prior_pconfs[i]);
	}
	if (waitKey(0) == 'q')
		exit(0);
}
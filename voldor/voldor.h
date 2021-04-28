#pragma once
#include "utils.h"
#include "config.h"
#include "geometry.h"
#include "../gpu-kernels/gpu_kernels.h"

enum OPTIMIZE_DEPTH_FLAG {
	OD_DEFAULT = 0,
	OD_ONLY_USE_DEPTH_PRIOR = 1,
	OD_UPDATE_RIGIDNESS_ONLY = 2,
};


class VOLDOR {
public:

	int n_flows, n_flows_init;
	int n_depth_priors;
	int w, h;
	int iters_cur;
	int iters_remain;
	Config cfg;
	bool has_disparity;

	Mat depth;
	vector<Mat> depth_priors;
	vector<Camera> depth_prior_poses;
	vector<Mat> depth_prior_pconfs;
	vector<Mat> depth_prior_confs;
	vector<Mat> flows;
	vector<Mat> rigidnesses;
	vector<Camera> cams;

	KittiGround ground;

	VOLDOR(Config cfg, bool exclusive_gpu_context = true) :
		cfg(cfg) {
		if (!cfg.silent)
			cfg.print_info();
	}

	void init(vector<Mat> _flows,
		Mat _disparity = Mat(), Mat _disparity_pconf = Mat(),
		vector<Mat> _depth_priors = vector<Mat>(),
		vector<Vec6f> _depth_prior_poses = vector<Vec6f>(),
		vector<Mat> _depth_prior_pconfs = vector<Mat>());

	int solve();


	void bootstrap();

	void optimize_cameras();

	void optimize_depth(OPTIMIZE_DEPTH_FLAG flag=OD_DEFAULT);

	void normalize_world_scale();

	void estimate_kitti_ground();

	void save_result(string save_dir);

	void debug();


private:
#if defined(WIN32) || defined(_WIN32)
	chrono::time_point<chrono::steady_clock> time_stamp;
#else
	chrono::system_clock::time_point time_stamp;
#endif
	void tic() {
		time_stamp = chrono::high_resolution_clock::now();
	}

	float toc() {
		return chrono::duration_cast<std::chrono::nanoseconds>(chrono::high_resolution_clock::now() - time_stamp).count() / 1e6;
	}
	void toc(string job_name) {
		cout << job_name << " elapsed time = " << toc() << "ms." << endl;
	}

};
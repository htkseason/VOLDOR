#pragma once
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include "ceres/ceres.h"
#include "glog/logging.h"
#include "align_frame_cost_fun.h"
#include "utils.h"

using namespace std;
using namespace cv;

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
	float vbf = 1000.f,
	float crw = 10.f,
	bool optimize_7dof = false,
	bool graduated_optimize = false,
	int stride = 4,
	float consistency_residual_bound = 1.f,
	bool debug = false);


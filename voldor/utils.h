#pragma once
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#define _USE_MATH_DEFINES

#if defined(WIN32) || defined(_WIN32) 
#define PATH_SEPARATOR '\\'
#else 
#define PATH_SEPARATOR '/'
#endif


#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>

using namespace cv;
using namespace std;

#define div_ceil(x, y) ( (x) / (y) + ((x) % (y) > 0) )


struct Camera {
	Mat F, E;

	Mat K = Mat::eye(3, 3, CV_32F);
	Mat K_inv = Mat::eye(3, 3, CV_32F);
	Mat R = Mat::eye(3, 3, CV_32F);
	Mat t = Mat::zeros(3, 1, CV_32F);
	Mat pose_covar = Mat::zeros(6, 6, CV_32F);
	//Mat _R2;
	//Mat pose_sample_mask;
	float pose_density = 0;
	int pose_sample_count = 0;
	float pose_rigidness_density = 0;
	int last_used_ms_iters = 0;
	int last_used_gu_iters = 0;

	Vec6f pose6() {
		Vec3f r = this->rvec();
		return Vec6f(r.val[0], r.val[1], r.val[2], t.at<float>(0), t.at<float>(1), t.at<float>(2));
	}

	Vec3f rvec() {
		Vec3f ret;
		Rodrigues(this->R, ret);
		return ret;
	}

	void save(FILE* fs) {
		Vec3f r = this->rvec();
		//           r1 r2 r3 t1 t2 t3
		fprintf(fs, "%f %f %f %f %f %f\n",
			r.val[0], r.val[1], r.val[2],
			t.at<float>(0), t.at<float>(1), t.at<float>(2));
	}

	void print_info() {
		cout << "pose pool size = " << this->pose_sample_count << endl;
		cout << "rigidness density = " << this->pose_rigidness_density << endl;
		cout << "pose density = " << this->pose_density << endl;
		cout << "pose covar mean scale = " << mean(pose_covar.diag())[0] << endl;;
		//cout << pose_covar.diag() << endl;
		cout << "last used meanshift iters = " << this->last_used_ms_iters << endl;
		cout << "last used gu iters = " << this->last_used_gu_iters << endl;
		cout << "pose trans mag = " << norm(this->t, NORM_L2) << endl;
		cout << "pose rot mag = " << norm(this->rvec(), NORM_L2) * 180 / 3.14159 << endl << endl;
	}
};


Mat vis_flow(Mat flow, float mag_scale = 0);

Mat load_flow(const char* file_path);


struct KittiGround {
	Vec3f normal = Vec3f(0, 0, 0);
	float height = 0;
	float confidence = 0;
	int used_iters = 0;
	float _height_median = 0;

	void save(FILE* fs) {
		fprintf(fs, "%f %f %f %f %f\n", this->height, this->normal.val[0], this->normal.val[1], this->normal.val[2], this->confidence);
	}

	void print_info() {
		cout << "ground height = " << this->height << endl;
		cout << "ground normal = " << this->normal << endl;
		cout << "ground confidence = " << this->confidence << endl;
		cout << "ground used iters = " << this->used_iters << endl;
		cout << "ground height median = " << this->_height_median << endl;
	}
};
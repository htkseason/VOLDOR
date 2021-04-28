#include "utils.h"

using namespace cv;
using namespace std;

Mat vis_flow(Mat flow, float mag_scale) {
	Mat flow_xy[2];
	Mat mag, angle;
	split(flow, flow_xy);
	cartToPolar(flow_xy[0], flow_xy[1], mag, angle, true);
	if (mag_scale <= 0)
		normalize(mag, mag, 0, 1, NORM_MINMAX);
	else
		mag /= mag_scale;
	Mat dst;
	std::vector<Mat> src{ angle, mag, Mat::ones(flow.size(), CV_32F) };
	merge(src, dst);
	cvtColor(dst, dst, COLOR_HSV2BGR);
	return dst;
}


Mat load_flow(const char* file_path) {
	FILE* fs = fopen(file_path, "rb");
	if (fs == NULL) {
		cout << file_path << " does not exist~!" << endl;
		throw;
	}

	float magic_num = 0;
	int w = 0, h = 0;
	fread(&magic_num, sizeof(float), 1, fs);
	assert(magic_num == 202021.25f);
	fread(&w, sizeof(int), 1, fs);
	fread(&h, sizeof(int), 1, fs);

	Mat flow(Size(w, h), CV_32FC2);
	fread(flow.data, sizeof(float), w*h * 2, fs);
	fclose(fs);
	return flow;
}

Mat rot_mat_3d(float degx, float degy, float degz) {
	degx /= 180 * 3.14159;
	degy /= 180 * 3.14159;
	degz /= 180 * 3.14159;
	Mat Rx = (Mat_<float>(3, 3) <<
		1, 0, 0,
		0, cosf(degx), -sinf(degx),
		0, sinf(degx), cosf(degx));
	Mat Ry = (Mat_<float>(3, 3) <<
		cosf(degy), 0, sinf(degy),
		0, 1, 0,
		-sinf(degy), 0, cosf(degy));
	Mat Rz = (Mat_<float>(3, 3) <<
		cosf(degz), -sinf(degz), 0,
		sinf(degz), cosf(degz), 0,
		0, 0, 1);

	return Rx * Ry * Rz;
}


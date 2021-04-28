#pragma once
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>


using namespace std;
using namespace cv;


template <typename T>
static T** get_mat_arr_pointer(vector<Mat> vector_mat) {
	T** ret = new T*[vector_mat.size()];
	for (int i = 0; i < vector_mat.size(); i++)
		ret[i] = (T*)vector_mat[i].data;
	return ret;
}

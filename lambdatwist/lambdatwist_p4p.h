#pragma once
#include "lambdatwist_p3p.h"


template <typename _T, typename T, int refinement_iterations = 5>
__mlib_host_device
bool lambdatwist_p4p(
	T* y1, T* y2, T* y3, T* y4,
	T* x1, T* x2, T* x3, T* x4,
	T fx, T fy, T cx, T cy,
	T R[3][3],
	T t[3]) {

	cvl::Vector3<_T> vy1((_T)((y1[0] - cx) / fx), (_T)((y1[1] - cy) / fy), _T(1.0));
	cvl::Vector3<_T> vy2((_T)((y2[0] - cx) / fx), (_T)((y2[1] - cy) / fy), _T(1.0));
	cvl::Vector3<_T> vy3((_T)((y3[0] - cx) / fx), (_T)((y3[1] - cy) / fy), _T(1.0));

	cvl::Vector3<_T> vx1((_T)(x1[0]), (_T)(x1[1]), (_T)(x1[2]));
	cvl::Vector3<_T> vx2((_T)(x2[0]), (_T)(x2[1]), (_T)(x2[2]));
	cvl::Vector3<_T> vx3((_T)(x3[0]), (_T)(x3[1]), (_T)(x3[2]));

	cvl::Vector<cvl::Matrix<_T, 3, 3>, 4> vRs;
	cvl::Vector<cvl::Vector3<_T>, 4> vTs;

	int n = cvl::p3p_lambdatwist<_T, refinement_iterations>(vy1, vy2, vy3, vx1, vx2, vx3, vRs, vTs);

	if (n == 0)
		return false;

	int ns = 0;
	_T min_reproj = 0;
	for (int i = 0; i < n; i++) {
		_T X3p = vRs(i)(0, 0) * x4[0] + vRs(i)(0, 1) * x4[1] + vRs(i)(0, 2) * x4[2] + vTs(i)(0);
		_T Y3p = vRs(i)(1, 0) * x4[0] + vRs(i)(1, 1) * x4[1] + vRs(i)(1, 2) * x4[2] + vTs(i)(1);
		_T Z3p = vRs(i)(2, 0) * x4[0] + vRs(i)(2, 1) * x4[1] + vRs(i)(2, 2) * x4[2] + vTs(i)(2);
		_T mu3p = cx + fx * X3p / Z3p;
		_T mv3p = cy + fy * Y3p / Z3p;
		_T reproj = (mu3p - y4[0])*(mu3p - y4[0]) + (mv3p - y4[1])*(mv3p - y4[1]);
		if (i == 0 || min_reproj > reproj) {
			ns = i;
			min_reproj = reproj;
		}
	}

	R[0][0] = (T)vRs(ns)(0, 0);
	R[0][1] = (T)vRs(ns)(0, 1);
	R[0][2] = (T)vRs(ns)(0, 2);
	R[1][0] = (T)vRs(ns)(1, 0);
	R[1][1] = (T)vRs(ns)(1, 1);
	R[1][2] = (T)vRs(ns)(1, 2);
	R[2][0] = (T)vRs(ns)(2, 0);
	R[2][1] = (T)vRs(ns)(2, 1);
	R[2][2] = (T)vRs(ns)(2, 2);
	t[0] = (T)vTs(ns)(0);
	t[1] = (T)vTs(ns)(1);
	t[2] = (T)vTs(ns)(2);


	//memcpy(R, Rs[ns], 3 * 3 * sizeof(T));
	//memcpy(t, ts[ns], 3 * sizeof(T));
	return true;
}
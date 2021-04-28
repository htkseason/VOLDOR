// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2016 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: vitus@google.com (Michael Vitus)

#ifndef EXAMPLES_CERES_TYPES_H_
#define EXAMPLES_CERES_TYPES_H_

#include <istream>
#include <map>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "ceres/rotation.h"


struct Pose3d {
	Eigen::Vector3d p;
	Eigen::Quaterniond q;
	double s;

	Pose3d() {}

	Pose3d(const double* pose7) {
		double quat[4];
		ceres::AngleAxisToQuaternion(pose7, quat);
		q.w() = quat[0];
		q.x() = quat[1];
		q.y() = quat[2];
		q.z() = quat[3];
		memcpy(p.data(), pose7 + 3, 3 * sizeof(double));
		s = pose7[6];
	}

	// The name of the data type in the g2o file format.
	static std::string name() { return "VERTEX_SE3:QUAT"; }

	Eigen::Matrix<double, 7, 1> to_pose7() {
		Eigen::Matrix<double, 7, 1> ret;
		double quat[4] = { q.w(), q.x(), q.y(), q.z() };
		ceres::QuaternionToAngleAxis(quat, ret.data());
		memcpy(ret.data() + 3, p.data(), 3 * sizeof(double));
		ret(6) = s;
		return ret;
	}

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// The constraint between two vertices in the pose graph. The constraint is the
// transformation from vertex id_begin to vertex id_end.
struct Constraint3d {
	int id_begin;
	int id_end;

	// The transformation that represents the pose of the end frame E w.r.t. the
	// begin frame B. In other words, it transforms a vector in the E frame to
	// the B frame.
	Pose3d t_be;

	// The inverse of the covariance matrix for the measurement. The order of the
	// entries are x, y, z, delta orientation.
	Eigen::Matrix<double, 7, 7> information;

	Constraint3d() {}

	Constraint3d(const int id1, const int id2, const double* pose7, const double* covar_rts) {
		id_begin = id1;
		id_end = id2;

		t_be = Pose3d(pose7);

		// angle-axis : axis*theta
		// quaternion : axis*sin(theta/2)
		// apply angle-axis to quaternion affine to covar
		const double theta_squared = pose7[0] * pose7[0] + pose7[1] * pose7[1] + pose7[2] * pose7[2];
		double k;
		if (theta_squared > 0.0) {
			const double theta = sqrt(theta_squared);
			k = sin(theta * 0.5) / theta;
		}
		else {
			k = 0.5;
		}

		Eigen::Matrix<double, 7, 7> covar; //change covar order to tvec/rvec/s with input rvec/tvec/s
		for (int i1 = 0; i1 < 7; i1++) {
			for (int i2 = 0; i2 < 7; i2++) {
				int i1_trs = i1 == 6 ? 6 : i1 < 3 ? i1 + 3 : i1 - 3;
				int i2_trs = i2 == 6 ? 6 : i2 < 3 ? i2 + 3 : i2 - 3;
				double s_q2r = 1;
				if (i1_trs < 3) s_q2r *= k;
				if (i2_trs < 3) s_q2r *= k;
				covar(i1, i2) = covar_rts[i1_trs * 7 + i2_trs] * s_q2r;
			}
		}

		information = covar.inverse().eval();
	}

	// The name of the data type in the g2o file format.
	static std::string name() { return "EDGE_SE3:QUAT"; }

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};



typedef std::vector<Constraint3d, Eigen::aligned_allocator<Constraint3d> >
VectorOfConstraints;

typedef std::map<int,
	Pose3d,
	std::less<int>,
	Eigen::aligned_allocator<std::pair<const int, Pose3d> > >
	MapOfPoses;

#endif  // EXAMPLES_CERES_TYPES_H_
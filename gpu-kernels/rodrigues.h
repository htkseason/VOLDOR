#include "utils.h"
#include "svd3_cuda.h"

// modified from Ceres/Rotation.h
__host__ __device__ static void RotationMatrixToAngleAxis(
	float R[3][3],
	float angle_axis[3]) {
	// x = k * 2 * sin(theta), where k is the axis of rotation.
	angle_axis[0] = R[2][1] - R[1][2];
	angle_axis[1] = R[0][2] - R[2][0];
	angle_axis[2] = R[1][0] - R[0][1];


	// Since the right hand side may give numbers just above 1.0 or
	// below -1.0 leading to atan misbehaving, we threshold.
	float costheta = fminf(fmaxf((
		R[0][0] + R[1][1] + R[2][2] - 1.f) * 0.5f,
		-1.f), 1.f);

	// sqrt is guaranteed to give non-negative results, so we only
	// threshold above.
	float sintheta = fminf(sqrtf(
		angle_axis[0] * angle_axis[0] +
		angle_axis[1] * angle_axis[1] +
		angle_axis[2] * angle_axis[2]) * 0.5f,
		1.f);

	// Use the arctan2 to get the right sign on theta
	const float theta = atan2f(sintheta, costheta);

	// Case 1: sin(theta) is large enough, so dividing by it is not a
	// problem. We do not use abs here, because while jets.h imports
	// std::abs into the namespace, here in this file, abs resolves to
	// the int version of the function, which returns zero always.
	//
	// We use a threshold much larger then the machine epsilon, because
	// if sin(theta) is small, not only do we risk overflow but even if
	// that does not occur, just dividing by a small number will result
	// in numerical garbage. So we play it safe.
	if ((sintheta > FLT_EPSILON) || (sintheta < -FLT_EPSILON)) {
		const float r = theta / (2.f * sintheta);
		angle_axis[0] *= r;
		angle_axis[1] *= r;
		angle_axis[2] *= r;
		return;
	}

	// Case 2: theta ~ 0, means sin(theta) ~ theta to a good
	// approximation.
	if (costheta > 0) {
		angle_axis[0] *= 0.5f;
		angle_axis[1] *= 0.5f;
		angle_axis[2] *= 0.5f;
		return;
	}

	// Case 3: theta ~ pi, this is the hard case. Since theta is large,
	// and sin(theta) is small. Dividing by theta by sin(theta) will
	// either give an overflow or worse still numerically meaningless
	// results. Thus we use an alternate more complicated formula
	// here.

	// Since cos(theta) is negative, division by (1-cos(theta)) cannot
	// overflow.
	const float inv_one_minus_costheta = 1.f / (1.f - costheta);

	// We now compute the absolute value of coordinates of the axis
	// vector using the diagonal entries of R. To resolve the sign of
	// these entries, we compare the sign of angle_axis[i]*sin(theta)
	// with the sign of sin(theta). If they are the same, then
	// angle_axis[i] should be positive, otherwise negative.
	for (int i = 0; i < 3; ++i) {
		angle_axis[i] = theta * sqrtf((R[i][i] - costheta) * inv_one_minus_costheta);
		if (((sintheta < 0) && (angle_axis[i] > 0)) ||
			((sintheta > 0) && (angle_axis[i] < 0))) {
			angle_axis[i] = -angle_axis[i];
		}
	}
}


__device__ static void rodrigues(float R[3][3], float rvec[3]) {
	float U[3][3], Vt[3][3], S[3];
	svd(R[0][0], R[0][1], R[0][2],
		R[1][0], R[1][1], R[1][2],
		R[2][0], R[2][1], R[2][2],

		U[0][0], U[0][1], U[0][2],
		U[1][0], U[1][1], U[1][2],
		U[2][0], U[2][1], U[2][2],

		S[0], S[1], S[2],

		Vt[0][0], Vt[1][0], Vt[2][0],
		Vt[0][1], Vt[1][1], Vt[2][1],
		Vt[0][2], Vt[1][2], Vt[2][2]
	);

	R[0][0] = U[0][0] * Vt[0][0] + U[0][1] * Vt[1][0] + U[0][2] * Vt[2][0];
	R[0][1] = U[0][0] * Vt[0][1] + U[0][1] * Vt[1][1] + U[0][2] * Vt[2][1];
	R[0][2] = U[0][0] * Vt[0][2] + U[0][1] * Vt[1][2] + U[0][2] * Vt[2][2];

	R[1][0] = U[1][0] * Vt[0][0] + U[1][1] * Vt[1][0] + U[1][2] * Vt[2][0];
	R[1][1] = U[1][0] * Vt[0][1] + U[1][1] * Vt[1][1] + U[1][2] * Vt[2][1];
	R[1][2] = U[1][0] * Vt[0][2] + U[1][1] * Vt[1][2] + U[1][2] * Vt[2][2];

	R[2][0] = U[2][0] * Vt[0][0] + U[2][1] * Vt[1][0] + U[2][2] * Vt[2][0];
	R[2][1] = U[2][0] * Vt[0][1] + U[2][1] * Vt[1][1] + U[2][2] * Vt[2][1];
	R[2][2] = U[2][0] * Vt[0][2] + U[2][1] * Vt[1][2] + U[2][2] * Vt[2][2];
	

	RotationMatrixToAngleAxis(R, rvec);

}
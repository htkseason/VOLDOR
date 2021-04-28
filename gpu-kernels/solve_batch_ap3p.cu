#include "utils.h"
#include "gpu_kernels.h"
#include "rodrigues.h"

#define N_THREADS 32

__constant__ static float _fx, _fy, _cx, _cy;

__host__ __device__ __inline__ static cuFloatComplex cuCsqrtf(cuFloatComplex x) {
	cuFloatComplex out;
	out.x = sqrtf(cuCabsf(x) * (x.x / cuCabsf(x) + 1.0f) / 2.0f);
	out.y = sqrtf(cuCabsf(x) * (1.0f - x.x / cuCabsf(x)) / 2.0f);
	out.y = -fabs(out.y);
	return out;
}

__host__ __device__ __inline__ static cuFloatComplex cuCpowf(const cuFloatComplex& z, float p) {
	float theta = atan2f(z.y, z.x);
	return make_cuFloatComplex((powf(cuCabsf(z), p) * cosf(p*theta)), (powf(cuCabsf(z), p)*sinf(p*theta)));
}


__host__ __device__ __inline__ static cuFloatComplex cuCnegf(cuFloatComplex x) {
	return make_cuFloatComplex(-x.x, -x.y);
}


__device__ static void solveQuartic(const float *factors, float *realRoots) {
	const float &a4 = factors[0];
	const float &a3 = factors[1];
	const float &a2 = factors[2];
	const float &a1 = factors[3];
	const float &a0 = factors[4];

	float a4_2 = a4 * a4;
	float a3_2 = a3 * a3;
	float a4_3 = a4_2 * a4;
	float a2a4 = a2 * a4;

	float p4 = (8 * a2a4 - 3 * a3_2) / (8 * a4_2);
	float q4 = (a3_2 * a3 - 4 * a2a4 * a3 + 8 * a1 * a4_2) / (8 * a4_3);
	float r4 = (256 * a0 * a4_3 - 3 * (a3_2 * a3_2) - 64 * a1 * a3 * a4_2 + 16 * a2a4 * a3_2) / (256 * (a4_3 * a4));

	float p3 = ((p4 * p4) / 12 + r4) / 3; // /=-3
	float q3 = (72 * r4 * p4 - 2 * p4 * p4 * p4 - 27 * q4 * q4) / 432; // /=2

	float t; // *=2

	cuFloatComplex w = make_cuFloatComplex(q3 * q3 - p3 * p3 * p3, 0);

	w = cuCsqrtf(w);
	if (q3 >= 0) {
		w.x = -w.x - q3;
		w.y = -w.y;
	}
	else {
		w = cuCsqrtf(w);
		w.x = w.x - q3;
	}

	if (w.y == 0.0f) {
		w.x = cbrtf(w.x);
		t = 2.0f * (w.x + p3 / w.x);
	}
	else {
		w = cuCpowf(w, (1.0f / 3.0f));
		t = 4.0f * w.x;
	}

	cuFloatComplex sqrt_2m = cuCsqrtf(make_cuFloatComplex(-2 * p4 / 3 + t, 0));
	float B_4A = -a3 / (4 * a4);
	cuFloatComplex complex1 = make_cuFloatComplex(4 * p4 / 3 + t, 0);
	cuFloatComplex complex2 = cuCdivf(make_cuFloatComplex(2 * q4, 0), sqrt_2m);

	float sqrt_2m_rh = sqrt_2m.x * 0.5f;
	float sqrt1 = cuCsqrtf(cuCnegf(cuCaddf(complex1, complex2))).x * 0.5f;
	realRoots[0] = B_4A + sqrt_2m_rh + sqrt1;
	realRoots[1] = B_4A + sqrt_2m_rh - sqrt1;
	float sqrt2 = cuCsqrtf(cuCnegf(cuCsubf(complex1, complex2))).x * 0.5f;
	realRoots[2] = B_4A - sqrt_2m_rh + sqrt2;
	realRoots[3] = B_4A - sqrt_2m_rh - sqrt2;
}


__host__ __device__ __inline__ static void polishQuarticRoots(const float *coeffs, float *roots) {
	const int iterations = 2;
	for (int i = 0; i < iterations; ++i) {
		for (int j = 0; j < 4; ++j) {
			float error =
				(((coeffs[0] * roots[j] + coeffs[1]) * roots[j] + coeffs[2]) * roots[j] + coeffs[3]) * roots[j] +
				coeffs[4];
			float
				derivative =
				((4 * coeffs[0] * roots[j] + 3 * coeffs[1]) * roots[j] + 2 * coeffs[2]) * roots[j] + coeffs[3];
			roots[j] -= error / derivative;
		}
	}
}

__host__ __device__ __inline__ static void vect_cross(const float *a, const float *b, float *result) {
	result[0] = a[1] * b[2] - a[2] * b[1];
	result[1] = -(a[0] * b[2] - a[2] * b[0]);
	result[2] = a[0] * b[1] - a[1] * b[0];
}

__host__ __device__ __inline__ static float vect_dot(const float *a, const float *b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__host__ __device__ __inline__ static float vect_norm(const float *a) {
	return sqrtf(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

__host__ __device__ __inline__ static void vect_scale(const float s, const float *a, float *result) {
	result[0] = a[0] * s;
	result[1] = a[1] * s;
	result[2] = a[2] * s;
}

__host__ __device__ __inline__ static void vect_sub(const float *a, const float *b, float *result) {
	result[0] = a[0] - b[0];
	result[1] = a[1] - b[1];
	result[2] = a[2] - b[2];
}

__host__ __device__ __inline__ static void vect_divide(const float *a, const float d, float *result) {
	result[0] = a[0] / d;
	result[1] = a[1] / d;
	result[2] = a[2] / d;
}

__host__ __device__ __inline__ static void mat_mult(const float a[3][3], const float b[3][3], float result[3][3]) {
	result[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0];
	result[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1];
	result[0][2] = a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2];

	result[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0];
	result[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1];
	result[1][2] = a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2];

	result[2][0] = a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0];
	result[2][1] = a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1];
	result[2][2] = a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2];
}

// This algorithm is from "Tong Ke, Stergios Roumeliotis, An Efficient Algebraic Solution to the Perspective-Three-Point Problem" (Accepted by CVPR 2017)
// See https://arxiv.org/pdf/1701.08237.pdf
// featureVectors: The 3 bearing measurements (normalized) stored as column vectors
// worldPoints: The positions of the 3 feature points stored as column vectors
// solutionsR: 4 possible solutions of rotation matrix of the world w.r.t the camera frame
// solutionsT: 4 possible solutions of translation of the world origin w.r.t the camera frame
__device__ static int computePoses(const float featureVectors[3][3],
	const float worldPoints[3][3],
	float solutionsR[4][3][3],
	float solutionsT[4][3]) {

	//world point vectors
	float w1[3] = { worldPoints[0][0], worldPoints[1][0], worldPoints[2][0] };
	float w2[3] = { worldPoints[0][1], worldPoints[1][1], worldPoints[2][1] };
	float w3[3] = { worldPoints[0][2], worldPoints[1][2], worldPoints[2][2] };
	// k1
	float u0[3];
	vect_sub(w1, w2, u0);

	float nu0 = vect_norm(u0);
	float k1[3];
	vect_divide(u0, nu0, k1);
	// bi
	float b1[3] = { featureVectors[0][0], featureVectors[1][0], featureVectors[2][0] };
	float b2[3] = { featureVectors[0][1], featureVectors[1][1], featureVectors[2][1] };
	float b3[3] = { featureVectors[0][2], featureVectors[1][2], featureVectors[2][2] };
	// k3,tz
	float k3[3];
	vect_cross(b1, b2, k3);
	float nk3 = vect_norm(k3);
	vect_divide(k3, nk3, k3);

	float tz[3];
	vect_cross(b1, k3, tz);
	// ui,vi
	float v1[3];
	vect_cross(b1, b3, v1);
	float v2[3];
	vect_cross(b2, b3, v2);

	float u1[3];
	vect_sub(w1, w3, u1);
	// coefficients related terms
	float u1k1 = vect_dot(u1, k1);
	float k3b3 = vect_dot(k3, b3);
	// f1i
	float f11 = k3b3;
	float f13 = vect_dot(k3, v1);
	float f15 = -u1k1 * f11;
	//delta
	float nl[3];
	vect_cross(u1, k1, nl);
	float delta = vect_norm(nl);
	vect_divide(nl, delta, nl);
	f11 *= delta;
	f13 *= delta;
	// f2i
	float u2k1 = u1k1 - nu0;
	float f21 = vect_dot(tz, v2);
	float f22 = nk3 * k3b3;
	float f23 = vect_dot(k3, v2);
	float f24 = u2k1 * f22;
	float f25 = -u2k1 * f21;
	f21 *= delta;
	f22 *= delta;
	f23 *= delta;
	float g1 = f13 * f22;
	float g2 = f13 * f25 - f15 * f23;
	float g3 = f11 * f23 - f13 * f21;
	float g4 = -f13 * f24;
	float g5 = f11 * f22;
	float g6 = f11 * f25 - f15 * f21;
	float g7 = -f15 * f24;
	float coeffs[5] = { g5 * g5 + g1 * g1 + g3 * g3,
						2 * (g5 * g6 + g1 * g2 + g3 * g4),
						g6 * g6 + 2 * g5 * g7 + g2 * g2 + g4 * g4 - g1 * g1 - g3 * g3,
						2 * (g6 * g7 - g1 * g2 - g3 * g4),
						g7 * g7 - g2 * g2 - g4 * g4 };
	float s[4];
	solveQuartic(coeffs, s);
	polishQuarticRoots(coeffs, s);

	float temp[3];
	vect_cross(k1, nl, temp);

	float Ck1nl[3][3] =
	{ {k1[0], nl[0], temp[0]},
	 {k1[1], nl[1], temp[1]},
	 {k1[2], nl[2], temp[2]} };

	float Cb1k3tzT[3][3] =
	{ {b1[0], b1[1], b1[2]},
	 {k3[0], k3[1], k3[2]},
	 {tz[0], tz[1], tz[2]} };

	float b3p[3];
	vect_scale((delta / k3b3), b3, b3p);

	int nb_solutions = 0;
	for (int i = 0; i < 4; ++i) {
		float ctheta1p = s[i];
		if (abs(ctheta1p) > 1)
			continue;
		float stheta1p = sqrt(1 - ctheta1p * ctheta1p);
		stheta1p = (k3b3 > 0) ? stheta1p : -stheta1p;
		float ctheta3 = g1 * ctheta1p + g2;
		float stheta3 = g3 * ctheta1p + g4;
		float ntheta3 = stheta1p / ((g5 * ctheta1p + g6) * ctheta1p + g7);
		ctheta3 *= ntheta3;
		stheta3 *= ntheta3;

		float C13[3][3] =
		{ {ctheta3,            0,         -stheta3},
		 {stheta1p * stheta3, ctheta1p,  stheta1p * ctheta3},
		 {ctheta1p * stheta3, -stheta1p, ctheta1p * ctheta3} };

		float temp_matrix[3][3];
		float R[3][3];
		mat_mult(Ck1nl, C13, temp_matrix);
		mat_mult(temp_matrix, Cb1k3tzT, R);

		// R' * p3
		float rp3[3] =
		{ w3[0] * R[0][0] + w3[1] * R[1][0] + w3[2] * R[2][0],
		 w3[0] * R[0][1] + w3[1] * R[1][1] + w3[2] * R[2][1],
		 w3[0] * R[0][2] + w3[1] * R[1][2] + w3[2] * R[2][2] };

		float pxstheta1p[3];
		vect_scale(stheta1p, b3p, pxstheta1p);

		vect_sub(pxstheta1p, rp3, solutionsT[nb_solutions]);

		solutionsR[nb_solutions][0][0] = R[0][0];
		solutionsR[nb_solutions][1][0] = R[0][1];
		solutionsR[nb_solutions][2][0] = R[0][2];
		solutionsR[nb_solutions][0][1] = R[1][0];
		solutionsR[nb_solutions][1][1] = R[1][1];
		solutionsR[nb_solutions][2][1] = R[1][2];
		solutionsR[nb_solutions][0][2] = R[2][0];
		solutionsR[nb_solutions][1][2] = R[2][1];
		solutionsR[nb_solutions][2][2] = R[2][2];

		nb_solutions++;
	}

	return nb_solutions;
}

__device__ static int solve_all(float R[4][3][3], float t[4][3], float mu0, float mv0, float X0, float Y0, float Z0, float mu1,
	float mv1, float X1, float Y1, float Z1, float mu2, float mv2, float X2, float Y2, float Z2) {
	float mk0, mk1, mk2;
	float norm;

	mu0 = (mu0 - _cx) / _fx;
	mv0 = (mv0 - _cy) / _fy;
	norm = sqrtf(mu0 * mu0 + mv0 * mv0 + 1);
	mk0 = 1.f / norm;
	mu0 *= mk0;
	mv0 *= mk0;

	mu1 = (mu1 - _cx) / _fx;
	mv1 = (mv1 - _cy) / _fy;
	norm = sqrtf(mu1 * mu1 + mv1 * mv1 + 1);
	mk1 = 1.f / norm;
	mu1 *= mk1;
	mv1 *= mk1;

	mu2 = (mu2 - _cx) / _fx;
	mv2 = (mv2 - _cy) / _fy;
	norm = sqrtf(mu2 * mu2 + mv2 * mv2 + 1);
	mk2 = 1.f / norm;
	mu2 *= mk2;
	mv2 *= mk2;

	float featureVectors[3][3] = { {mu0, mu1, mu2},
								   {mv0, mv1, mv2},
								   {mk0, mk1, mk2} };
	float worldPoints[3][3] = { {X0, X1, X2},
								{Y0, Y1, Y2},
								{Z0, Z1, Z2} };

	return computePoses(featureVectors, worldPoints, R, t);
}


__global__ static void solve(float* d_p2s, float* d_p3s, float* d_rvecs, float* d_tvecs, curandState* d_rand_states, int N_pts, int N_poses) {

	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx < N_poses) {
		int i1 = curand_uniform(&d_rand_states[idx])*N_pts;
		int i2 = curand_uniform(&d_rand_states[idx])*N_pts;
		int i3 = curand_uniform(&d_rand_states[idx])*N_pts;
		int i4 = curand_uniform(&d_rand_states[idx])*N_pts;
		float Rs[4][3][3], ts[4][3];

		int n = solve_all(Rs, ts,
			d_p2s[i1 * 2 + 0], d_p2s[i1 * 2 + 1],
			d_p3s[i1 * 3 + 0], d_p3s[i1 * 3 + 1], d_p3s[i1 * 3 + 2],

			d_p2s[i2 * 2 + 0], d_p2s[i2 * 2 + 1],
			d_p3s[i2 * 3 + 0], d_p3s[i2 * 3 + 1], d_p3s[i2 * 3 + 2],

			d_p2s[i3 * 2 + 0], d_p2s[i3 * 2 + 1],
			d_p3s[i3 * 3 + 0], d_p3s[i3 * 3 + 1], d_p3s[i3 * 3 + 2]);


		if (n == 0) {
			d_rvecs[idx * 3 + 0] = CUDART_NAN_F; d_rvecs[idx * 3 + 1] = CUDART_NAN_F; d_rvecs[idx * 3 + 2] = CUDART_NAN_F;
			d_tvecs[idx * 3 + 0] = CUDART_NAN_F; d_tvecs[idx * 3 + 1] = CUDART_NAN_F; d_tvecs[idx * 3 + 2] = CUDART_NAN_F;
			return; // false
		}

		int ns = 0;
		float min_reproj = 0;
		for (int i = 0; i < n; i++) {
			float X3p = Rs[i][0][0] * d_p3s[i4 * 3 + 0] + Rs[i][0][1] * d_p3s[i4 * 3 + 1] + Rs[i][0][2] * d_p3s[i4 * 3 + 2] + ts[i][0];
			float Y3p = Rs[i][1][0] * d_p3s[i4 * 3 + 0] + Rs[i][1][1] * d_p3s[i4 * 3 + 1] + Rs[i][1][2] * d_p3s[i4 * 3 + 2] + ts[i][1];
			float Z3p = Rs[i][2][0] * d_p3s[i4 * 3 + 0] + Rs[i][2][1] * d_p3s[i4 * 3 + 1] + Rs[i][2][2] * d_p3s[i4 * 3 + 2] + ts[i][2];
			float mu3p = _cx + _fx * X3p / Z3p;
			float mv3p = _cy + _fy * Y3p / Z3p;
			float reproj = SQR(mu3p - d_p2s[i4 * 2 + 0]) + SQR(mv3p - d_p2s[i4 * 2 + 1]);
			if (i == 0 || min_reproj > reproj) {
				ns = i;
				min_reproj = reproj;
			}
		}
		rodrigues(Rs[ns], &d_rvecs[idx * 3]);
		d_tvecs[idx * 3 + 0] = ts[ns][0];
		d_tvecs[idx * 3 + 1] = ts[ns][1];
		d_tvecs[idx * 3 + 2] = ts[ns][2];
	}
}

__global__ static void init_rand_states(curandState* d_rand_states, int N) {
	int idx = blockDim.x*blockIdx.x + threadIdx.x;
	if (idx < N)
		curand_init(RAND_SEED, idx, 0, &d_rand_states[idx]);
}


int solve_batch_p3p_ap3p_gpu(float* h_p3s, float* h_p2s, float* h_o_rvecs, float* h_o_tvecs, float* h_K, int N_pts, int N_poses) {

	// copy K to gpu constant memory
	static float cache_symbols[4] = { 0 };
	if (h_K) {
		CUDA_UPDATE_SYMBOL_IF_CHANGED(h_K[0], cache_symbols[0], _fx);
		CUDA_UPDATE_SYMBOL_IF_CHANGED(h_K[2], cache_symbols[2], _cx);
		CUDA_UPDATE_SYMBOL_IF_CHANGED(h_K[4], cache_symbols[1], _fy);
		CUDA_UPDATE_SYMBOL_IF_CHANGED(h_K[5], cache_symbols[3], _cy);
	}

	// allocate and copy pts2+pts3
	float* d_p2s;
	float* d_p3s;
	cudaMalloc((void**)&d_p2s, N_pts * 2 * sizeof(float));
	cudaMalloc((void**)&d_p3s, N_pts * 3 * sizeof(float));
	cudaMemcpy(d_p2s, h_p2s, N_pts * 2 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_p3s, h_p3s, N_pts * 3 * sizeof(float), cudaMemcpyHostToDevice);
	gpuErrchk;

	// allocate device memory for R,t
	float* d_rvecs;
	float* d_tvecs;
	cudaMalloc((void**)&d_rvecs, N_poses * 3 * sizeof(float));
	cudaMalloc((void**)&d_tvecs, N_poses * 3 * sizeof(float));
	gpuErrchk;

	// init rand seeds
	curandState* d_rand_states;
	cudaMalloc((void**)&d_rand_states, N_poses * sizeof(curandState));
	init_rand_states << <DIV_CEIL(N_poses, N_THREADS), N_THREADS >> > (d_rand_states, N_poses);

	// solve batch ap3p
	solve << <DIV_CEIL(N_poses, N_THREADS), N_THREADS >> > (d_p2s, d_p3s, d_rvecs, d_tvecs, d_rand_states, N_pts, N_poses);
	gpuErrchk;

	// copy back R,t
	cudaMemcpy(h_o_rvecs, d_rvecs, N_poses * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_o_tvecs, d_tvecs, N_poses * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	gpuErrchk;

	// free all
	cudaFree(d_rvecs);
	cudaFree(d_tvecs);
	cudaFree(d_p2s);
	cudaFree(d_p3s);
	cudaFree(d_rand_states);
	gpuErrchk;

	return cudaSuccess;
}

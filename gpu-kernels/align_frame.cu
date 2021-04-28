#include "utils.h"
#include "gpu_kernels.h"
#include "gblur.h"
#include "vops.h"
#include "gmat.h"

#define MAX_FRAMES 64
#define BLOCK_WIDTH 16
#define N_PARAMS 9

__constant__ static float _fx, _cx, _fy, _cy;
__constant__ static float _fxi, _cxi, _fyi, _cyi; // inv intrinsic
//__constant__ static float _poses[MAX_FRAMES][N_PARAMS];
__constant__ static float _params_ref[N_PARAMS], _params_tar[N_PARAMS];
__constant__ static float _vbf, _crw;
__constant__ static int _N, _w, _h;

static int N_, w_, h_;
static bool use_photo_consistency;


__constant__ static GMatf _d_images;
__constant__ static GMatf2 _d_dimages;

__constant__ static GMatf _d_depths;
__constant__ static GMatf2 _d_ddepths;

__constant__ static GMatf4 _d_normals;

__constant__ static GMatf _d_residual;
__constant__ static GMatf _d_jacobian;
__constant__ static GMatf _d_weights;


static GMatf d_images;
static GMatf2 d_dimages;

static GMatf d_depths;
static GMatf2 d_ddepths;

static GMatf4 d_normals;

static GMatf d_residual;
static GMatf d_jacobian;
static GMatf d_weights;

__host__ __device__ static float3 rot_with_rvec(float3 p3, float3 rvec, float jacobian_over_rvec[3][3] = NULL, float jacobian_over_p3[3][3] = NULL) {
	float theta2 = vnorm2(rvec);
	if (theta2 > FLT_EPSILON) {
		float theta = sqrt(theta2);
		float cos_theta = cos(theta);
		float sin_theta = sin(theta);
		float theta_inv = 1.f / theta;
		float3 w = vmul(rvec, theta_inv);
		float3 w_cross_pt = vcross(w, p3);
		float tmp = vdot(w, p3) * (1.0f - cos_theta);
		if (jacobian_over_p3) {
			jacobian_over_p3[0][0] = cos_theta - ((rvec.x*rvec.x)*(cos_theta - 1)) / theta2;
			jacobian_over_p3[0][1] = -(rvec.z*sin_theta) / theta - (rvec.x*rvec.y*(cos_theta - 1)) / theta2;
			jacobian_over_p3[0][2] = (rvec.y*sin_theta) / theta - (rvec.x*rvec.z*(cos_theta - 1)) / theta2;

			jacobian_over_p3[1][0] = (rvec.z*sin_theta) / theta - (rvec.x*rvec.y*(cos_theta - 1)) / theta2;
			jacobian_over_p3[1][1] = cos_theta - ((rvec.y*rvec.y)*(cos_theta - 1)) / theta2;
			jacobian_over_p3[1][2] = -(rvec.x*sin_theta) / theta - (rvec.y*rvec.z*(cos_theta - 1)) / theta2;

			jacobian_over_p3[2][0] = -(rvec.y*sin_theta) / theta - (rvec.x*rvec.z*(cos_theta - 1)) / theta2;
			jacobian_over_p3[2][1] = (rvec.x*sin_theta) / theta - (rvec.y*rvec.z*(cos_theta - 1)) / theta2;
			jacobian_over_p3[2][2] = cos_theta - ((rvec.z*rvec.z)*(cos_theta - 1)) / theta2;
		}
		if (jacobian_over_rvec) {
			float theta3_2 = sqrt(theta2*theta);
			jacobian_over_rvec[0][0] = sin_theta * ((rvec.x*rvec.z*p3.y) / theta3_2 - (rvec.x*rvec.y*p3.z) / theta3_2) - ((cos_theta - 1)*((rvec.x*p3.x) / theta + (rvec.y*p3.y) / theta + (rvec.z*p3.z) / theta)) / theta + (rvec.x*(cos_theta - 1)*(((rvec.x*rvec.x)*p3.x) / theta3_2 - p3.x / theta + (rvec.x*rvec.y*p3.y) / theta3_2 + (rvec.x*rvec.z*p3.z) / theta3_2)) / theta - (rvec.x*p3.x*sin_theta) / theta + ((rvec.x*rvec.x)*(cos_theta - 1)*((rvec.x*p3.x) / theta + (rvec.y*p3.y) / theta + (rvec.z*p3.z) / theta)) / theta3_2 + ((rvec.x*rvec.x)*sin_theta*((rvec.x*p3.x) / theta + (rvec.y*p3.y) / theta + (rvec.z*p3.z) / theta)) / theta2 - (rvec.x*cos_theta*((rvec.z*p3.y) / theta - (rvec.y*p3.z) / theta)) / theta;
			jacobian_over_rvec[0][1] = sin_theta * (p3.z / theta - ((rvec.y*rvec.y)*p3.z) / theta3_2 + (rvec.y*rvec.z*p3.y) / theta3_2) + (rvec.x*(cos_theta - 1)*(((rvec.y*rvec.y)*p3.y) / theta3_2 - p3.y / theta + (rvec.x*rvec.y*p3.x) / theta3_2 + (rvec.y*rvec.z*p3.z) / theta3_2)) / theta - (rvec.y*p3.x*sin_theta) / theta - (rvec.y*cos_theta*((rvec.z*p3.y) / theta - (rvec.y*p3.z) / theta)) / theta + (rvec.x*rvec.y*(cos_theta - 1)*((rvec.x*p3.x) / theta + (rvec.y*p3.y) / theta + (rvec.z*p3.z) / theta)) / theta3_2 + (rvec.x*rvec.y*sin_theta*((rvec.x*p3.x) / theta + (rvec.y*p3.y) / theta + (rvec.z*p3.z) / theta)) / theta2;
			jacobian_over_rvec[0][2] = (rvec.x*(cos_theta - 1)*(((rvec.z*rvec.z)*p3.z) / theta3_2 - p3.z / theta + (rvec.x*rvec.z*p3.x) / theta3_2 + (rvec.y*rvec.z*p3.y) / theta3_2)) / theta - sin_theta * (p3.y / theta - ((rvec.z*rvec.z)*p3.y) / theta3_2 + (rvec.y*rvec.z*p3.z) / theta3_2) - (rvec.z*p3.x*sin_theta) / theta - (rvec.z*cos_theta*((rvec.z*p3.y) / theta - (rvec.y*p3.z) / theta)) / theta + (rvec.x*rvec.z*(cos_theta - 1)*((rvec.x*p3.x) / theta + (rvec.y*p3.y) / theta + (rvec.z*p3.z) / theta)) / theta3_2 + (rvec.x*rvec.z*sin_theta*((rvec.x*p3.x) / theta + (rvec.y*p3.y) / theta + (rvec.z*p3.z) / theta)) / theta2;

			jacobian_over_rvec[1][0] = (rvec.y*(cos_theta - 1)*(((rvec.x*rvec.x)*p3.x) / theta3_2 - p3.x / theta + (rvec.x*rvec.y*p3.y) / theta3_2 + (rvec.x*rvec.z*p3.z) / theta3_2)) / theta - sin_theta * (p3.z / theta - ((rvec.x*rvec.x)*p3.z) / theta3_2 + (rvec.x*rvec.z*p3.x) / theta3_2) - (rvec.x*p3.y*sin_theta) / theta + (rvec.x*cos_theta*((rvec.z*p3.x) / theta - (rvec.x*p3.z) / theta)) / theta + (rvec.x*rvec.y*(cos_theta - 1)*((rvec.x*p3.x) / theta + (rvec.y*p3.y) / theta + (rvec.z*p3.z) / theta)) / theta3_2 + (rvec.x*rvec.y*sin_theta*((rvec.x*p3.x) / theta + (rvec.y*p3.y) / theta + (rvec.z*p3.z) / theta)) / theta2;
			jacobian_over_rvec[1][1] = (rvec.y*(cos_theta - 1)*(((rvec.y*rvec.y)*p3.y) / theta3_2 - p3.y / theta + (rvec.x*rvec.y*p3.x) / theta3_2 + (rvec.y*rvec.z*p3.z) / theta3_2)) / theta - ((cos_theta - 1)*((rvec.x*p3.x) / theta + (rvec.y*p3.y) / theta + (rvec.z*p3.z) / theta)) / theta - sin_theta * ((rvec.y*rvec.z*p3.x) / theta3_2 - (rvec.x*rvec.y*p3.z) / theta3_2) - (rvec.y*p3.y*sin_theta) / theta + ((rvec.y*rvec.y)*(cos_theta - 1)*((rvec.x*p3.x) / theta + (rvec.y*p3.y) / theta + (rvec.z*p3.z) / theta)) / theta3_2 + ((rvec.y*rvec.y)*sin_theta*((rvec.x*p3.x) / theta + (rvec.y*p3.y) / theta + (rvec.z*p3.z) / theta)) / theta2 + (rvec.y*cos_theta*((rvec.z*p3.x) / theta - (rvec.x*p3.z) / theta)) / theta;
			jacobian_over_rvec[1][2] = sin_theta * (p3.x / theta - ((rvec.z*rvec.z)*p3.x) / theta3_2 + (rvec.x*rvec.z*p3.z) / theta3_2) + (rvec.y*(cos_theta - 1)*(((rvec.z*rvec.z)*p3.z) / theta3_2 - p3.z / theta + (rvec.x*rvec.z*p3.x) / theta3_2 + (rvec.y*rvec.z*p3.y) / theta3_2)) / theta - (rvec.z*p3.y*sin_theta) / theta + (rvec.z*cos_theta*((rvec.z*p3.x) / theta - (rvec.x*p3.z) / theta)) / theta + (rvec.y*rvec.z*(cos_theta - 1)*((rvec.x*p3.x) / theta + (rvec.y*p3.y) / theta + (rvec.z*p3.z) / theta)) / theta3_2 + (rvec.y*rvec.z*sin_theta*((rvec.x*p3.x) / theta + (rvec.y*p3.y) / theta + (rvec.z*p3.z) / theta)) / theta2;

			jacobian_over_rvec[2][0] = sin_theta * (p3.y / theta - ((rvec.x*rvec.x)*p3.y) / theta3_2 + (rvec.x*rvec.y*p3.x) / theta3_2) + (rvec.z*(cos_theta - 1)*(((rvec.x*rvec.x)*p3.x) / theta3_2 - p3.x / theta + (rvec.x*rvec.y*p3.y) / theta3_2 + (rvec.x*rvec.z*p3.z) / theta3_2)) / theta - (rvec.x*p3.z*sin_theta) / theta - (rvec.x*cos_theta*((rvec.y*p3.x) / theta - (rvec.x*p3.y) / theta)) / theta + (rvec.x*rvec.z*(cos_theta - 1)*((rvec.x*p3.x) / theta + (rvec.y*p3.y) / theta + (rvec.z*p3.z) / theta)) / theta3_2 + (rvec.x*rvec.z*sin_theta*((rvec.x*p3.x) / theta + (rvec.y*p3.y) / theta + (rvec.z*p3.z) / theta)) / theta2;
			jacobian_over_rvec[2][1] = (rvec.z*(cos_theta - 1)*(((rvec.y*rvec.y)*p3.y) / theta3_2 - p3.y / theta + (rvec.x*rvec.y*p3.x) / theta3_2 + (rvec.y*rvec.z*p3.z) / theta3_2)) / theta - sin_theta * (p3.x / theta - ((rvec.y*rvec.y)*p3.x) / theta3_2 + (rvec.x*rvec.y*p3.y) / theta3_2) - (rvec.y*p3.z*sin_theta) / theta - (rvec.y*cos_theta*((rvec.y*p3.x) / theta - (rvec.x*p3.y) / theta)) / theta + (rvec.y*rvec.z*(cos_theta - 1)*((rvec.x*p3.x) / theta + (rvec.y*p3.y) / theta + (rvec.z*p3.z) / theta)) / theta3_2 + (rvec.y*rvec.z*sin_theta*((rvec.x*p3.x) / theta + (rvec.y*p3.y) / theta + (rvec.z*p3.z) / theta)) / theta2;
			jacobian_over_rvec[2][2] = sin_theta * ((rvec.y*rvec.z*p3.x) / theta3_2 - (rvec.x*rvec.z*p3.y) / theta3_2) - ((cos_theta - 1)*((rvec.x*p3.x) / theta + (rvec.y*p3.y) / theta + (rvec.z*p3.z) / theta)) / theta + (rvec.z*(cos_theta - 1)*(((rvec.z*rvec.z)*p3.z) / theta3_2 - p3.z / theta + (rvec.x*rvec.z*p3.x) / theta3_2 + (rvec.y*rvec.z*p3.y) / theta3_2)) / theta - (rvec.z*p3.z*sin_theta) / theta + ((rvec.z*rvec.z)*(cos_theta - 1)*((rvec.x*p3.x) / theta + (rvec.y*p3.y) / theta + (rvec.z*p3.z) / theta)) / theta3_2 + ((rvec.z*rvec.z)*sin_theta*((rvec.x*p3.x) / theta + (rvec.y*p3.y) / theta + (rvec.z*p3.z) / theta)) / theta2 - (rvec.z*cos_theta*((rvec.y*p3.x) / theta - (rvec.x*p3.y) / theta)) / theta;

		}
		return vadd(vadd(
			vmul(p3, cos_theta),
			vmul(w_cross_pt, sin_theta)),
			vmul(w, tmp));
	}
	else {
		float3 w_cross_pt = vcross(rvec, p3);

		if (jacobian_over_p3) {
			jacobian_over_p3[0][0] = 1;
			jacobian_over_p3[0][1] = -rvec.z;
			jacobian_over_p3[0][2] = rvec.y;

			jacobian_over_p3[1][0] = rvec.z;
			jacobian_over_p3[1][1] = 1;
			jacobian_over_p3[1][2] = -rvec.x;

			jacobian_over_p3[2][0] = -rvec.y;
			jacobian_over_p3[2][1] = rvec.x;
			jacobian_over_p3[2][2] = 1;
		}
		if (jacobian_over_rvec) {
			jacobian_over_rvec[0][0] = 0;
			jacobian_over_rvec[0][1] = p3.z;
			jacobian_over_rvec[0][2] = -p3.y;

			jacobian_over_rvec[1][0] = -p3.z;
			jacobian_over_rvec[1][1] = 0;
			jacobian_over_rvec[1][2] = p3.x;

			jacobian_over_rvec[2][0] = p3.y;
			jacobian_over_rvec[2][1] = -p3.x;
			jacobian_over_rvec[2][2] = 0;

		}
		return vadd(p3, w_cross_pt);
	}
}


__device__ __inline__ static float3 proj_p2_to_p3(float2 p2, float depth, float jacobian_over_depth[3] = NULL) {
	if (jacobian_over_depth) {
		jacobian_over_depth[0] = (_fxi * p2.x + _cxi);
		jacobian_over_depth[1] = (_fyi * p2.y + _cyi);
		jacobian_over_depth[2] = 1;
	}
	return make_float3(
		(_fxi * p2.x + _cxi) * depth,
		(_fyi * p2.y + _cyi) * depth,
		depth);
}

__device__ __inline__ static float2 proj_p3_to_p2(float3 p3, float jacobian[2][3] = NULL) {
	if (jacobian) {
		jacobian[0][0] = _fx / p3.z;
		jacobian[0][1] = 0;
		jacobian[0][2] = -(_fx*p3.x) / (p3.z*p3.z);
		jacobian[1][0] = 0;
		jacobian[1][1] = _fy / p3.z;
		jacobian[1][2] = -(_fy*p3.y) / (p3.z*p3.z);
	}
	return make_float2(
		(_fx * p3.x) / p3.z + _cx,
		(_fy * p3.y) / p3.z + _cy);
}



__global__ static void init_normal_ddepth() {
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int f = blockDim.z*blockIdx.z + threadIdx.z;
	if (x < _w && y < _h) {
		// compute normal
		float3 p3t = proj_p2_to_p3(make_float2(x, y - 1), _d_depths.at_safe(x, y - 1, f));
		float3 p3b = proj_p2_to_p3(make_float2(x, y + 1), _d_depths.at_safe(x, y + 1, f));
		float3 p3l = proj_p2_to_p3(make_float2(x - 1, y), _d_depths.at_safe(x - 1, y, f));
		float3 p3r = proj_p2_to_p3(make_float2(x + 1, y), _d_depths.at_safe(x + 1, y, f));

		float3 nvec = vcross(vsub(p3t, p3b), vsub(p3l, p3r));
		nvec = vdiv(nvec, vnorm(nvec));

		float3 p3 = proj_p2_to_p3(make_float2(x, y), 1.f);

		if (vdot(p3, nvec) > 0)
			nvec = vneg(nvec); // make normal point to view point

		_d_normals.at(x, y, f) = make_float4(nvec.x, nvec.y, nvec.z, 0);

		// ddepth
		_d_ddepths.at(x, y, f) = make_float2(
			0.3f*(_d_depths.at_safe(x + 1, y, f) - _d_depths.at_safe(x - 1, y, f)) +
			0.1f*(_d_depths.at_safe(x + 1, y - 1, f) - _d_depths.at_safe(x - 1, y - 1, f)) +
			0.1f*(_d_depths.at_safe(x + 1, y + 1, f) - _d_depths.at_safe(x - 1, y + 1, f)),

			0.3f*(_d_depths.at_safe(x, y + 1, f) - _d_depths.at_safe(x, y - 1, f)) +
			0.1f*(_d_depths.at_safe(x - 1, y + 1, f) - _d_depths.at_safe(x - 1, y - 1, f)) +
			0.1f*(_d_depths.at_safe(x + 1, y + 1, f) - _d_depths.at_safe(x + 1, y - 1, f))
		);
	}
}

__global__ static void init_dimage() {
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int f = blockDim.z*blockIdx.z + threadIdx.z;
	if (x < _w && y < _h) {
		// dimage
		_d_dimages.at(x, y, f) = make_float2(
			0.3f*(_d_images.at_safe(x + 1, y, f) - _d_images.at_safe(x - 1, y, f)) +
			0.1f*(_d_images.at_safe(x + 1, y - 1, f) - _d_images.at_safe(x - 1, y - 1, f)) +
			0.1f*(_d_images.at_safe(x + 1, y + 1, f) - _d_images.at_safe(x - 1, y + 1, f)),

			0.3f*(_d_images.at_safe(x, y + 1, f) - _d_images.at_safe(x, y - 1, f)) +
			0.1f*(_d_images.at_safe(x - 1, y + 1, f) - _d_images.at_safe(x - 1, y - 1, f)) +
			0.1f*(_d_images.at_safe(x + 1, y + 1, f) - _d_images.at_safe(x + 1, y - 1, f))
		);
	}
}

__global__ static void compute_residual(int f_ref, int f_tar, bool use_photo_consistency, bool compute_jacobian) {
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;

	if (x < _w && y < _h) {

		// poses are [R|t] cam->world
		float3 rvec = make_float3(_params_ref[0], _params_ref[1], _params_ref[2]);
		float3 tvec = make_float3(_params_ref[3], _params_ref[4], _params_ref[5]);
		float d_scale_ref = _params_ref[6]; // depth scale of ref frame
		float c_scale_ref = _params_ref[7]; // color scale of ref frame
		float c_offset_ref = _params_ref[8]; // color offset of ref frame


		// p3 reference before scaling
		float3 __p3r__p2r_d;
		float p2r_d_bs = _d_depths.at(x, y, f_ref);
		float p2r_d = p2r_d_bs * expf(d_scale_ref);
		float3 p3r = proj_p2_to_p3(make_float2(x, y), p2r_d, compute_jacobian ? (float*)&__p3r__p2r_d : NULL);

		float __p2r_d__d_scale_ref = p2r_d;


		float __p3w__rvec[3][3];
		float __p3w__p3r[3][3];
		float3 p3w = vadd(rot_with_rvec(p3r, rvec, compute_jacobian ? __p3w__rvec : NULL, compute_jacobian ? __p3w__p3r : NULL), tvec);

		float3 rvec0 = make_float3(_params_tar[0], _params_tar[1], _params_tar[2]);
		float3 tvec0 = make_float3(_params_tar[3], _params_tar[4], _params_tar[5]);
		rvec0 = vneg(rvec0);  // make pose world->cam
		tvec0 = vneg(rot_with_rvec(tvec0, rvec0));
		float d_scale_tar = _params_tar[6]; // depth offset of tar frame
		float c_scale_tar = _params_tar[7]; // color scale of tar frame
		float c_offset_tar = _params_tar[8]; // color offset of tar frame


		float __p3t__p3w[3][3];
		float3 p3t = vadd(rot_with_rvec(p3w, rvec0, NULL, compute_jacobian ? __p3t__p3w : NULL), tvec0);

		// p2 at target frame
		float __p2t__p3t[2][3];
		float2 p2t = proj_p3_to_p2(p3t, compute_jacobian ? __p2t__p3t : NULL);

		if (p2t.x < 0 || p2t.x >= _w || p2t.y < 0 || p2t.y >= _h ||
			p3t.z < 1.f) {
			_d_residual.at(x, y) = CUDART_NAN_F;
			return;
		}

		// depth of p2t
		float p2t_d_bs = _d_depths.at_tex(p2t.x, p2t.y, f_tar);
		float p2t_d = p2t_d_bs * expf(d_scale_tar);

		// target p3/p2 of normal plane
		float3 nvec_tar = f4_to_f3(_d_normals.at_tex(p2t.x, p2t.y, f_tar));

		float3 p3t_tar_ray = vmul(p3t, p2t_d / p3t.z);
		float3 p3t_diff_geo = vmul(nvec_tar, vdot(nvec_tar, vsub(p3t_tar_ray, p3t))); // tar->ref
		float3 p3t_tar_geo = vadd(p3t, p3t_diff_geo);
		float2 p2t_tar_geo = proj_p3_to_p2(p3t_tar_geo);
		//p2t_tar_geo = p2t;
		if (p2t_tar_geo.x < 0 || p2t_tar_geo.x >= _w || p2t_tar_geo.y < 0 || p2t_tar_geo.y >= _h) {
			_d_residual.at(x, y) = CUDART_NAN_F;
			return;
		}
		float residual_depth = 0.5f * vnorm2(p3t_diff_geo);
		float drw = SQR(_vbf / (fmaxf(p3t_tar_geo.z, 1.0f) * fmaxf(p3t.z, 1.0f)));


		// compute color residual  ====================================
		float c_ref = 0;
		float c_tar_bs = 0;
		float c_tar = 0;
		float residual_color = 0;
		float __c_tar__c_scale_ref = 0;
		float __c_ref__c_offset_ref = 0;

		if (use_photo_consistency) {
			c_ref = _d_images.at(x, y, f_ref) + c_offset_ref;
			c_tar_bs = _d_images.at_tex(p2t.x, p2t.y, f_tar) + c_offset_tar;
			c_tar = c_tar_bs * (expf(c_scale_ref) / expf(c_scale_tar));
			residual_color = 0.5f * (c_ref - c_tar)*(c_ref - c_tar);

			__c_tar__c_scale_ref = c_tar;
			__c_ref__c_offset_ref = 1.f;
		}




		// depth residual ===============================================
		if (use_photo_consistency) {
			_d_residual.at(x, y) =
				drw * residual_depth +
				_crw * residual_color;
		}
		else {
			_d_residual.at(x, y) = drw * residual_depth;
		}

		// if jacobian
		if (compute_jacobian) {

			float __residual__residual_depth = drw;
			float __residual__residual_color = _crw;


			/* ============== residual_depth --> p3t ============== */
			float3 __residual_depth__p3t = vneg(p3t_diff_geo);

			/* ============================================================ */

			/* ============== residual_color --> p3t / c_scale_ref / c_offset_ref ============== */
			float3 __residual_color__p3t = make_float3(0, 0, 0);
			float __residual_color__c_scale_ref = 0;
			float __residual_color__c_offset_ref = 0;
			if (use_photo_consistency) {
				float __residual_color__c_tar = c_tar - c_ref;
				float __residual_color__c_ref = c_ref - c_tar;

				float2 __c_tar__p2t = _d_dimages.at_tex(p2t.x, p2t.y, f_tar);
				float2 __residual_color__p2t = vmul(__c_tar__p2t, __residual_color__c_tar);
				//float3 __residual_color__p3t;
				mm_12x23((float*)&__residual_color__p2t, __p2t__p3t, (float*)&__residual_color__p3t);

				__residual_color__c_scale_ref = __residual_color__c_tar * __c_tar__c_scale_ref;
				__residual_color__c_offset_ref = __residual_color__c_ref * __c_ref__c_offset_ref;
			}

			/* ============================================================ */


			/* ============== p3t --> rvec/tvec/ds ============== */
			// p3t --> p3w (already obtained, 3x3)

			// p3w --> rvec (already obtained, 3x3)

			// p3w --> tvec (identity, 3x3)

			float3 __residual__p3t =
				vadd(vmul(__residual_depth__p3t, __residual__residual_depth),
					vmul(__residual_color__p3t, __residual__residual_color));

			float __residual__p3w[3];
			mm_13x33((float*)&__residual__p3t, __p3t__p3w, __residual__p3w);

			// __p3w__tvec is identity
			float* __residual__tvec = __residual__p3w;

			float __residual__rvec[3];
			mm_13x33(__residual__p3w, __p3w__rvec, __residual__rvec);

			float3 __residual__p3r;
			mm_13x33(__residual__p3w, __p3w__p3r, (float*)&__residual__p3r);
			float __residual__d_scale_ref = vdot(__residual__p3r, __p3r__p2r_d) * __p2r_d__d_scale_ref;

			// 6dof pose
			_d_jacobian.at(x*N_PARAMS + 0, y) = __residual__rvec[0];
			_d_jacobian.at(x*N_PARAMS + 1, y) = __residual__rvec[1];
			_d_jacobian.at(x*N_PARAMS + 2, y) = __residual__rvec[2];
			_d_jacobian.at(x*N_PARAMS + 3, y) = __residual__tvec[0];
			_d_jacobian.at(x*N_PARAMS + 4, y) = __residual__tvec[1];
			_d_jacobian.at(x*N_PARAMS + 5, y) = __residual__tvec[2];
			// depth correction
			_d_jacobian.at(x*N_PARAMS + 6, y) = __residual__d_scale_ref;
			// color correction
			if (use_photo_consistency) {
				_d_jacobian.at(x*N_PARAMS + 7, y) = __residual__residual_color * __residual_color__c_scale_ref;
				_d_jacobian.at(x*N_PARAMS + 8, y) = __residual__residual_color * __residual_color__c_offset_ref;
			}
			else {
				_d_jacobian.at(x*N_PARAMS + 7, y) = 0;
				_d_jacobian.at(x*N_PARAMS + 8, y) = 0;
			}
		}
	}
}

__global__ void apply_weighted_sqrt_cauchy_loss(int ref_fid, bool apply_weights, bool apply_to_jacobian) {
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;

	if (x < _w && y < _h) {
		float weight = 1.f;
		if (apply_weights)
			weight = _d_weights.at(x, y, ref_fid);
		float residual2 = weight * _d_residual.at(x, y);
		if (residual2 > FLT_EPSILON) {

			float loss = log(residual2 + 1.f);
			float __loss__residual2 = 1.f / (residual2 + 1.f);

			float sqrt_loss = sqrt(loss);
			float __sqrt_loss__loss = 0.5f / (sqrt_loss);


			_d_residual.at(x, y) = sqrt_loss;
			if (apply_to_jacobian) {
				//if (isfinite(__sqrt_loss__loss))
				for (int i = 0; i < N_PARAMS; i++)
					_d_jacobian.at(x*N_PARAMS + i, y) *= (__sqrt_loss__loss * __loss__residual2 * weight);

			}
		}
	}

}


int align_frame_eval_gpu(
	int ref_fid,
	int tar_fid,
	const float* h_params_ref,
	const float* h_params_tar,
	float* h_o_residual, float* h_o_jacobian,
	bool apply_weights) {
	const dim3 block_size(BLOCK_WIDTH, BLOCK_WIDTH, 1);
	const dim3 grid_size(DIV_CEIL(w_, BLOCK_WIDTH), DIV_CEIL(h_, BLOCK_WIDTH), 1);

	if (h_params_ref)
		cudaMemcpyToSymbol(_params_ref, h_params_ref, N_PARAMS * sizeof(float));
	if (h_params_tar)
		cudaMemcpyToSymbol(_params_tar, h_params_tar, N_PARAMS * sizeof(float));
	d_residual.zeros();
	d_jacobian.zeros();
	gpuErrchk;


	compute_residual << <grid_size, block_size >> > (ref_fid, tar_fid, use_photo_consistency, h_o_jacobian != NULL);
	gpuErrchk;
	apply_weighted_sqrt_cauchy_loss << <grid_size, block_size >> > (ref_fid, apply_weights, h_o_jacobian != NULL);
	gpuErrchk;

	if (h_o_residual)
		d_residual.copy_to_host(h_o_residual, make_cudaPos(0, 0, 0), w_, h_, 1);
	if (h_o_jacobian)
		d_jacobian.copy_to_host(h_o_jacobian, make_cudaPos(0, 0, 0), w_*N_PARAMS, h_, 1);
	gpuErrchk;

	return cudaSuccess;
}

int align_frame_init_gpu(
	float* h_images[],
	float* h_depths[],
	float* h_weights[],
	float* h_K,
	float vbf, float crw,
	int N, int w, int h) {

	N_ = N;
	w_ = w;
	h_ = h;

	const dim3 block_size(BLOCK_WIDTH, BLOCK_WIDTH, 1);
	const dim3 grid_size(DIV_CEIL(w, BLOCK_WIDTH), DIV_CEIL(h, BLOCK_WIDTH), N);

	cudaMemcpyToSymbol(_vbf, &vbf, sizeof(float));
	cudaMemcpyToSymbol(_crw, &crw, sizeof(float));
	cudaMemcpyToSymbol(_N, &N, sizeof(int));
	cudaMemcpyToSymbol(_w, &w, sizeof(int));
	cudaMemcpyToSymbol(_h, &h, sizeof(int));
	gpuErrchk;

	// copy images
	if (h_images && crw > 0) {
		if (d_images.create(w, h, N)) {
			d_images.bind_tex();
			cudaMemcpyToSymbol(_d_images, &d_images, sizeof(GMatf));
		}
		gpuErrchk;
		for (int i = 0; i < N; i++)
			d_images.copy_from_host(h_images[i], make_cudaPos(0, 0, i), w, h, 1);
		gpuErrchk;
		use_photo_consistency = true;
	}
	else {
		use_photo_consistency = false;
	}

	// copy depths
	if (d_depths.create(w, h, N)) {
		d_depths.bind_tex();
		cudaMemcpyToSymbol(_d_depths, &d_depths, sizeof(GMatf));
	}
	gpuErrchk;
	for (int i = 0; i < N; i++)
		d_depths.copy_from_host(h_depths[i], make_cudaPos(0, 0, i), w, h, 1);

	// copy weights
	if (d_weights.create(w, h, N)) {
		//d_weights.bind_tex();
		cudaMemcpyToSymbol(_d_weights, &d_weights, sizeof(GMatf));
	}
	gpuErrchk;
	for (int i = 0; i < N; i++)
		d_weights.copy_from_host(h_weights[i], make_cudaPos(0, 0, i), w, h, 1);

	// allocate other mats
	if (use_photo_consistency) {
		if (d_dimages.create(w, h, N)) {
			d_dimages.bind_tex();
			cudaMemcpyToSymbol(_d_dimages, &d_dimages, sizeof(GMatf2));
		}
		gpuErrchk;
	}

	if (d_ddepths.create(w, h, N)) {
		d_ddepths.bind_tex();
		cudaMemcpyToSymbol(_d_ddepths, &d_ddepths, sizeof(GMatf2));
	}
	gpuErrchk;

	if (d_normals.create(w, h, N)) {
		d_normals.bind_tex();
		cudaMemcpyToSymbol(_d_normals, &d_normals, sizeof(GMatf4));
	}
	gpuErrchk;

	if (d_residual.create(w, h, 1))
		cudaMemcpyToSymbol(_d_residual, &d_residual, sizeof(GMatf));
	gpuErrchk;

	if (d_jacobian.create(w*N_PARAMS, h, 1))
		cudaMemcpyToSymbol(_d_jacobian, &d_jacobian, sizeof(GMatf));
	gpuErrchk;


	float h_K4_inv[4]{ 1.f / h_K[0], -h_K[2] / h_K[0], 1.f / h_K[4], -h_K[5] / h_K[4] };

	cudaMemcpyToSymbol(_fx, &h_K[0], sizeof(float));
	cudaMemcpyToSymbol(_cx, &h_K[2], sizeof(float));
	cudaMemcpyToSymbol(_fy, &h_K[4], sizeof(float));
	cudaMemcpyToSymbol(_cy, &h_K[5], sizeof(float));
	gpuErrchk;

	cudaMemcpyToSymbol(_fxi, &h_K4_inv[0], sizeof(float));
	cudaMemcpyToSymbol(_cxi, &h_K4_inv[1], sizeof(float));
	cudaMemcpyToSymbol(_fyi, &h_K4_inv[2], sizeof(float));
	cudaMemcpyToSymbol(_cyi, &h_K4_inv[3], sizeof(float));
	gpuErrchk;


	init_normal_ddepth << <grid_size, block_size >> > ();
	if (use_photo_consistency)
		init_dimage << <grid_size, block_size >> > ();
	gpuErrchk;

	return cudaSuccess;
}


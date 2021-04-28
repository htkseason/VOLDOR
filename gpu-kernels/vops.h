#pragma once
#include "utils.h"

__device__ __host__ __inline__ static float3 f4_to_f3(const float4 f4) {
	return make_float3(f4.x, f4.y, f4.z);
}

__device__ __host__ __inline__ static float4 f3_to_f4(const float3 f3) {
	return make_float4(f3.x, f3.y, f3.z, 0.f);
}

__device__ __host__ __inline__ static void mm_33x33(const float a[3][3], const float b[3][3], float ret[3][3]) {
	ret[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0];
	ret[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1];
	ret[0][2] = a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2];

	ret[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0];
	ret[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1];
	ret[1][2] = a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2];

	ret[2][0] = a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0];
	ret[2][1] = a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1];
	ret[2][2] = a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2];
}

__device__ __host__ __inline__ static void mm_13x33(const float a[3], const float b[3][3], float ret[3]) {
	ret[0] = a[0] * b[0][0] + a[1] * b[1][0] + a[2] * b[2][0];
	ret[1] = a[0] * b[0][1] + a[1] * b[1][1] + a[2] * b[2][1];
	ret[2] = a[0] * b[0][2] + a[1] * b[1][2] + a[2] * b[2][2];
}

__device__ __host__ __inline__ static void mm_33x31(const float a[3][3], const float b[3], float ret[3]) {
	ret[0] = a[0][0] * b[0] + a[0][1] * b[1] + a[0][2] * b[2];
	ret[1] = a[1][0] * b[0] + a[1][1] * b[1] + a[1][2] * b[2];
	ret[2] = a[2][0] * b[0] + a[2][1] * b[1] + a[2][2] * b[2];
}

__device__ __host__ __inline__ static void mm_12x23(const float a[2], const float b[2][3], float ret[3]) {
	ret[0] = a[0] * b[0][0] + a[1] * b[1][0];
	ret[1] = a[0] * b[0][1] + a[1] * b[1][1];
	ret[2] = a[0] * b[0][2] + a[1] * b[1][2];
}

__device__ __host__ __inline__ static float3 vcross(float3 v1, float3 v2) {
	return make_float3(
		v1.y * v2.z - v1.z * v2.y,
		v1.z * v2.x - v1.x * v2.z,
		v1.x * v2.y - v1.y * v2.x);
}

__device__ __host__ __inline__ static float vdot(float3 v1, float3 v2) {
	return  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__device__ __host__ __inline__ static float vnorm(float3 v) {
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ __host__ __inline__ static float vnorm2(float3 v) {
	return v.x * v.x + v.y * v.y + v.z * v.z;
}

__device__ __host__ __inline__ static float3 vsub(float3 v1, float3 v2) {
	return make_float3(
		v1.x - v2.x,
		v1.y - v2.y,
		v1.z - v2.z);
}
__device__ __host__ __inline__ static float3 vadd(float3 v1, float3 v2) {
	return make_float3(
		v1.x + v2.x,
		v1.y + v2.y,
		v1.z + v2.z);
}

__device__ __host__ __inline__ static float3 vdiv(float3 v1, float s) {
	return make_float3(
		v1.x / s,
		v1.y / s,
		v1.z / s);
}

__device__ __host__ __inline__ static float3 vdiv(float3 v1, float3 v2) {
	return make_float3(
		v1.x / v2.x,
		v1.y / v2.y,
		v1.z / v2.z);
}

__device__ __host__ __inline__ static float3 vmul(float3 v1, float s) {
	return make_float3(
		v1.x * s,
		v1.y * s,
		v1.z * s);
}

__device__ __host__ __inline__ static float3 vmul(float3 v1, float3 v2) {
	return make_float3(
		v1.x * v2.x,
		v1.y * v2.y,
		v1.z * v2.z);
}

__device__ __host__ __inline__ static float2 vmul(float2 v1, float s) {
	return make_float2(
		v1.x * s,
		v1.y * s);
}

__device__ __host__ __inline__ static float3 vneg(float3 v1) {
	return make_float3(
		-v1.x,
		-v1.y,
		-v1.z);
}
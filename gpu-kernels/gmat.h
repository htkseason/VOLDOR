#pragma once
#include "utils.h"

template <typename T>
class GMat {
private:
	size_t _width = 0, _height = 0, _depth = 0;
	//cudaArray* _tex_arr = NULL;

public:
	cudaPitchedPtr _dptr = { 0 };
	cudaTextureObject_t _tex_obj = NULL;

	__host__ __device__ __inline__ size_t width() { return _width; }
	__host__ __device__ __inline__ size_t height() { return _height; }
	__host__ __device__ __inline__ size_t depth() { return _depth; }

	// allocate new mat if neccessary (return 1), otherwise, return 0.
	int create(const size_t width, const size_t height, const size_t depth, bool lazy_depth = false) {
		if ((width == _width && height == _height && depth == _depth) ||
			(lazy_depth && width == _width && height == _height && depth <= _depth))
			return 0;
		free();
		//printf("%d x %d x %d mat created \n", (int)width, (int)height, (int)depth);
		cudaMalloc3D(&_dptr, make_cudaExtent(width * sizeof(T), height, depth));

		_width = width;
		_height = height;
		_depth = depth;
		//gpuErrchk;
		return 1;
	}

	int zeros() {
		return cudaMemset3D(_dptr, 0, make_cudaExtent(_width * sizeof(T), _height, _depth));
	}


	int bind_tex() {
		if (_tex_obj)
			return 0;

		// create texture object
		cudaResourceDesc res_desc;
		memset(&res_desc, 0, sizeof(res_desc));
		res_desc.resType = cudaResourceTypePitch2D;
		res_desc.res.pitch2D.desc = cudaCreateChannelDesc<T>();
		res_desc.res.pitch2D.devPtr = _dptr.ptr;
		res_desc.res.pitch2D.height = _depth * _height;
		res_desc.res.pitch2D.width = _width;
		res_desc.res.pitch2D.pitchInBytes = _dptr.pitch;

		cudaTextureDesc tex_desc;
		memset(&tex_desc, 0, sizeof(tex_desc));
		tex_desc.addressMode[0] = cudaAddressModeClamp;
		tex_desc.addressMode[1] = cudaAddressModeClamp;
		tex_desc.addressMode[2] = cudaAddressModeClamp;
		tex_desc.filterMode = cudaFilterModeLinear;
		tex_desc.readMode = cudaReadModeElementType;
		tex_desc.normalizedCoords = false;

		cudaCreateTextureObject(&_tex_obj, &res_desc, &tex_desc, NULL);
		//gpuErrchk;

		return 1;
	}

	__host__ __device__ __inline__ bool empty() {
		return _width == 0 || _height == 0 || _depth == 0;
	}

	int free() {
		_width = 0;
		_height = 0;
		_depth = 0;

		if (_tex_obj) {
			cudaDestroyTextureObject(_tex_obj);
			_tex_obj = NULL;
		}

		if (_dptr.ptr) {
			cudaFree(_dptr.ptr);
			memset(&_dptr, 0, sizeof(cudaPitchedPtr));
		}

		//gpuErrchk;
		return cudaSuccess;
	}

	int copy_from_device(
		const cudaPitchedPtr src, const cudaPos src_pos,
		const cudaPos dst_pos,
		const size_t width, const size_t height, const size_t depth) {
		cudaMemcpy3DParms parm;
		memset(&parm, 0, sizeof(parm));
		parm.kind = cudaMemcpyDeviceToDevice;

		parm.dstPtr = _dptr;
		parm.dstPos = dst_pos;
		parm.extent = make_cudaExtent(width * sizeof(T), height, depth);

		parm.srcPtr = src;
		parm.srcPos = src_pos;

		cudaMemcpy3D(&parm);
		//gpuErrchk;

		return cudaSuccess;
	}

	int copy_to_device(
		const cudaPitchedPtr dst, const cudaPos dst_pos,
		const cudaPos src_pos,
		const size_t width, const size_t height, const size_t depth) {
		cudaMemcpy3DParms parm;
		memset(&parm, 0, sizeof(parm));
		parm.kind = cudaMemcpyDeviceToDevice;

		parm.srcPtr = _dptr;
		parm.srcPos = src_pos;
		parm.extent = make_cudaExtent(width * sizeof(T), height, depth);

		parm.dstPtr = dst;
		parm.dstPos = dst_pos;

		cudaMemcpy3D(&parm);
		//gpuErrchk;

		return cudaSuccess;
	}

	int copy_from_host(const T* src, const cudaPos dst_pos,
		const size_t width, const size_t height, const size_t depth) {
		cudaMemcpy3DParms parm;
		memset(&parm, 0, sizeof(parm));
		parm.kind = cudaMemcpyHostToDevice;

		parm.dstPtr = _dptr;
		parm.dstPos = dst_pos;
		parm.extent = make_cudaExtent(width * sizeof(T), height, depth);

		parm.srcPtr = make_cudaPitchedPtr((void*)src, width * sizeof(T), width, height);
		parm.srcPos = make_cudaPos(0, 0, 0);

		cudaMemcpy3D(&parm);
		//gpuErrchk;

		return cudaSuccess;
	}

	int copy_to_host(const T* dst, const cudaPos src_pos,
		const size_t width, const size_t height, const size_t depth) {
		cudaMemcpy3DParms parm;
		memset(&parm, 0, sizeof(parm));
		parm.kind = cudaMemcpyDeviceToHost;

		parm.srcPtr = _dptr;
		parm.srcPos = src_pos;
		parm.extent = make_cudaExtent(width * sizeof(T), height, depth);

		parm.dstPtr = make_cudaPitchedPtr((void*)dst, width * sizeof(T), width, height);
		parm.dstPos = make_cudaPos(0, 0, 0);

		cudaMemcpy3D(&parm);
		//gpuErrchk;

		return cudaSuccess;
	}

	__device__ __inline__ T& at(const size_t x, const size_t y, const size_t d = 0) {
		return *((T*)(((char*)_dptr.ptr) + d * _dptr.ysize * _dptr.pitch + y * _dptr.pitch) + x);
	}

	__device__ __inline__ T at_tex(const float x, const float y, const int d = 0) {
		// probably may have boundary issues due to 2d vs 2d-layered texture
		// better create texture from 2d-layered array
		return tex2D<T>(_tex_obj, x + 0.5f, d*_height + y + 0.5f);
	}

	__device__ __inline__ T& at_safe(const size_t x, const size_t y, const size_t d = 0) {
		return *((T*)(((char*)_dptr.ptr) +
			max(min(d, (size_t)(_depth - 1)), (size_t)0) * _dptr.ysize * _dptr.pitch +
			max(min(y, (size_t)(_height - 1)), (size_t)0) * _dptr.pitch) +
			max(min(x, (size_t)(_width - 1)), (size_t)0));
	}

	__device__ __inline__ T at_tex_safe(const float x, const float y, const int d = 0) {
		// probably may have boundary issues due to 2d vs 2d-layered texture
		// better create texture from 2d-layered array
		return tex2D<T>(_tex_obj,
			max(min(x, (float)(_width - 1)), (float)0) + 0.5f,
			max(min(d, (int)(_depth - 1)), (int)0)*_height +
			max(min(y, (float)(_height - 1)), (float)0) + 0.5f);
	}

};


typedef GMat<float> GMatf;
typedef GMat<float2> GMatf2;
typedef GMat<float3> GMatf3;
typedef GMat<float4> GMatf4;
typedef GMat<curandState> GMatRnd;



/*
#include <stdio.h>
typedef GMat<float, false> GMatf;
typedef GMat<float, true> GMatf_tex;


__global__ void test_tex(GMatf_tex mat) {
	printf("%d,%d,%d\n", (int)mat.width(),(int)mat.height(), (int)mat.depth());
	for (int d = 0; d < 2; d++) {
		for (int y = 0; y < 2; y++) {
			for (int x = 0; x < 3; x++) {
				printf("%f, ", mat.at_tex(x, y, d));
			}
			printf("\n");
		}
		printf("\n");
	}
}


__global__ void test(GMatf mat) {
	printf("%d,%d,%d\n", (int)mat.width(), (int)mat.height(), (int)mat.depth());
	for (int d = 0; d < 2; d++) {
		for (int y = 0; y < 2; y++) {
			for (int x = 0; x < 3; x++) {
				printf("%f, ", mat.at(x, y, d));
			}
			printf("\n");
		}
		printf("\n");
	}
	mat.at(0, 0, 0) = 233;
}

int main() {
	float hdata[] = { 1,2,3,4,5,6, 7,8,9,10,11,12 };

	GMatf_tex mat_tex;
	mat_tex.create(1, 1, 1);
	mat_tex.create(3, 2, 2);
	mat_tex.copy_from_host(hdata, make_cudaPos(0, 0, 0), 3, 2, 2);
	mat_tex.bind_tex();
	test_tex << <1, 1 >> > (mat_tex);
	mat_tex.free();


	GMatf mat;
	mat.create(1, 1, 1);
	mat.create(3, 2, 2);
	mat.copy_from_host(hdata, make_cudaPos(0, 0, 0), 3, 2, 1);
	mat.copy_from_host(hdata, make_cudaPos(0, 0, 1), 3, 2, 1);
	test << <1, 1 >> > (mat);
	test << <1, 1 >> > (mat);
	mat.free();

	return 0;
}
*/
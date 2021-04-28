// A simple but not efficient code for device gblur ...
#include "gblur.h"

#define GBLUR_MAXIMUM_HALF_KWIDTH 128
#define GBLUR_BLOCK_WIDTH 32
#define GBLUR_HORIZONTAL 0
#define GBLUR_VERTICAL 1

__constant__ static float _gkernel[GBLUR_MAXIMUM_HALF_KWIDTH];
__constant__ static int _gkernel_hw;

template<int type>
__global__ static void gblur_kernel(GMatf src, GMatf dst) {
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int d = blockDim.z*blockIdx.z + threadIdx.z;
	if (x < src.width() && y < src.height()) {
		float sum = _gkernel[0] * src.at(x, y, d);
		float sum_w = _gkernel[0];
		for (int k = 1; k < _gkernel_hw; k++) {
			if (type == GBLUR_HORIZONTAL) {
				if (x + k < src.width()) {
					sum += _gkernel[k] * src.at(x + k, y, d);
					sum_w += _gkernel[k];
				}
				if (x - k >= 0) {
					sum += _gkernel[k] * src.at(x - k, y, d);
					sum_w += _gkernel[k];
				}
			}
			if (type == GBLUR_VERTICAL) {
				if (y + k < src.height()) {
					sum += _gkernel[k] * src.at(x, y + k, d);
					sum_w += _gkernel[k];
				}
				if (y - k >= 0) {
					sum += _gkernel[k] * src.at(x, y - k, d);
					sum_w += _gkernel[k];
				}
			}
		}
		dst.at(x, y, d) = sum / sum_w;
	}
}


int gblur_gpu(GMatf src, GMatf& dst, float sigma, int ksize) {
	const dim3 block_size(GBLUR_BLOCK_WIDTH, GBLUR_BLOCK_WIDTH, 1);
	const dim3 grid_size(DIV_CEIL(src.width(), GBLUR_BLOCK_WIDTH),
		DIV_CEIL(src.height(), GBLUR_BLOCK_WIDTH),
		src.depth());

	if (ksize == 0)
		ksize = max((int)ceil(6 * sigma), 3);
	int half_ksize = ksize / 2 + 1;
	if (half_ksize > GBLUR_MAXIMUM_HALF_KWIDTH)
		return cudaErrorInvalidFilterSetting;

	float ht_gkernel[GBLUR_MAXIMUM_HALF_KWIDTH];
	for (int i = 0; i < half_ksize; i++)
		ht_gkernel[i] = expf(-(float)(i*i) / (float)(2 * sigma*sigma));
	cudaMemcpyToSymbol(_gkernel, ht_gkernel, GBLUR_MAXIMUM_HALF_KWIDTH * sizeof(float));
	cudaMemcpyToSymbol(_gkernel_hw, &half_ksize, sizeof(int));

	GMatf dst_tmp;
	dst_tmp.create(src.width(), src.height(), src.depth());
	gblur_kernel<GBLUR_VERTICAL> << < grid_size, block_size >> > (src, dst_tmp);
	dst.create(src.width(), src.height(), src.depth());
	gblur_kernel<GBLUR_HORIZONTAL> << < grid_size, block_size >> > (dst_tmp, dst);
	dst_tmp.free();
	return cudaSuccess;
}
#pragma once
#ifndef  __CUDACC__
#define __CUDACC__
#endif

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"
#include "cuComplex.h"
#include "math_constants.h"
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define RAND_SEED 233
#define ZDE FLT_EPSILON

#define GPU_ERR_CHECK
#ifdef GPU_ERR_CHECK
#define gpuErrchk  { cudaError_t code = cudaGetLastError(); if (code!= cudaSuccess) {printf("GPUassert : %s\n%s at line %d\n", cudaGetErrorString(code), __FILE__, __LINE__); return code;} }
#else
#define gpuErrchk
#endif

#define DIV_CEIL(x, y) (y==0 ? 0: (x) / (y) + ((x) % (y) > 0) )
#define SQR(x) ( (x)*(x) )
#define CUBE(x) ( (x)*(x)*(x) )
#define L2_NORM_SQR(x,y) ( (x)*(x) + (y)*(y) )
#define L2_NORM(x,y) ( sqrtf( (x)*(x) + (y)*(y) ) )

#define CUDA_UPDATE_SYMBOL_IF_CHANGED(var, cache, symbol) if (var!=cache) {cudaMemcpyToSymbol(symbol, &var, sizeof var); cache=var;}
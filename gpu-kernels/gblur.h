// A simple but not efficient code for device gblur ...
#pragma once
#include "utils.h"
#include "gmat.h"

int gblur_gpu(GMatf src, GMatf& dst, float sigma, int ksize = 0);
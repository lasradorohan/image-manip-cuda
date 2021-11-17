#ifndef BLACKWHITE
#define BLACKWHITE

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include "device_launch_parameters.h"
#include "Dispatch.h"

void executeBlackWhite(uchar4* image, size_t height, size_t width);


#endif
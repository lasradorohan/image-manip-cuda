#ifndef BLACKWHITE
#define BLACKWHITE

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include "device_launch_parameters.h"
#include "Dispatch.h"

class ImageCommand {
public:
	virtual void execute(uchar4** image, size_t* height, size_t* width) = 0;
	virtual ~ImageCommand();
};

class BlackWhiteImageCommand : public ImageCommand {
public:
	void execute(uchar4** image, size_t* height, size_t* width);
};

class RotateImageCommand : public ImageCommand {
	float phi;
public:
	RotateImageCommand(float phi);
	void execute(uchar4** image, size_t* height, size_t* width);
};

class GammaCorrectionImageCommand : public ImageCommand {
	float gc;
public:
	GammaCorrectionImageCommand(float gc);
	void execute(uchar4** image, size_t* height, size_t* width);
};

class RadialDistortionImageCommand : public ImageCommand {
	float k1;
public:
	RadialDistortionImageCommand(float phi);
	void execute(uchar4** image, size_t* height, size_t* width);
};

void executeContrast(uchar4** image, size_t* height, size_t* width, float alpha);
#endif
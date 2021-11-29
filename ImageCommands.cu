#include "kernels.h"

ImageCommand::~ImageCommand() {};


__global__ void blackWhite(uchar4* image, size_t height, size_t width) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < height && y < width) {
		int idx = x * width + y;
		unsigned char val = 0.299 * image[idx].x + 0.587 * image[idx].y + 0.114 * image[idx].z;
		image[idx].x = val;
		image[idx].y = val;
		image[idx].z = val;
	}
}

void BlackWhiteImageCommand::execute(uchar4** image, size_t* height, size_t* width) {
	size_t in_h = *height;
	size_t in_w = *width;
	uchar4* d_image;
	cudaMalloc(&d_image, in_h * in_w * sizeof(uchar4));
	cudaMemcpy(d_image, *image, in_h * in_w * sizeof(uchar4), cudaMemcpyHostToDevice);
	blackWhite <<<dim3(1 + ((in_h - 1) / 32), 1 + ((in_w - 1) / 32), 1), dim3(32, 32, 1) >>> (d_image, in_h, in_w);
	cudaMemcpy(*image, d_image, in_h * in_w * sizeof(uchar4), cudaMemcpyDeviceToHost);
	//image memory, height and width dont change
}

std::string BlackWhiteImageCommand::toString() {
	return "BlackWhite()";
}


__global__ void rotate(uchar4* image_in, uchar4* image_out,
	size_t in_h, size_t in_w, size_t out_h, size_t out_w, float phi) {
	int out_x = blockIdx.x * blockDim.x + threadIdx.x;
	int out_y = blockIdx.y * blockDim.y + threadIdx.y;
	int in_x = (out_x - out_w / 2.f) * __cosf(phi) - (out_y - out_h / 2.f) * __sinf(phi) + in_w/2.f;
	int in_y = (out_y - out_h / 2.f) * __cosf(phi) + (out_x - out_w / 2.f) * __sinf(phi) + in_h/2.f;
	if (out_x < out_w && out_y < out_h) {
		if (0 <= in_x && in_x < in_w && 0 <= in_y && in_y < in_h) {
			image_out[out_w * out_y + out_x] = image_in[in_y * in_w + in_x];
		}
		/*else {
			image_out[out_w * out_y + out_x] = {0, 0, 0, 0};
		}*/
	}
}

RotateImageCommand::RotateImageCommand(float phi) : phi(phi * PI / 180.0f) {}

void RotateImageCommand::execute(uchar4** image, size_t* height, size_t* width) {
	size_t in_h = *height;
	size_t in_w = *width;
	size_t out_w = in_w * cos(phi) + in_h * sin(phi);
	size_t out_h = in_w * sin(phi) + in_h * cos(phi);
	uchar4* image_out = new uchar4[out_w * out_h];
	uchar4* d_in, * d_out;
	cudaMalloc(&d_in, in_h * in_w * sizeof(uchar4));
	cudaMalloc(&d_out, out_h * out_w * sizeof(uchar4));
	cudaMemcpy(d_in, *image, in_h * in_w * sizeof(uchar4), cudaMemcpyHostToDevice);
	rotate <<< dim3(1 + ((out_w - 1) / 32), 1 + ((out_h - 1) / 32), 1), dim3(32, 32, 1) >>> (d_in, d_out, in_h, in_w, out_h, out_w, phi);
	cudaMemcpy(image_out, d_out, out_h * out_w * sizeof(uchar4), cudaMemcpyDeviceToHost);
	delete[] * image;
	*image = image_out;
	*height = out_h;
	*width = out_w;
}

std::string RotateImageCommand::toString() {
	return "Rotate(" + std::to_string(phi) + ")";
}

__global__ void gammaCorrection(uchar4* image, size_t height, size_t width, float gc) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < height && y < width) {
		int idx = x * width + y;
		float r, g, b;
		r = image[idx].x;
		g = image[idx].y;
		b = image[idx].z;
		r = 255 * __powf((r / 255), (1 / gc));
		g = 255 * __powf((g / 255), (1 / gc));
		b = 255 * __powf((b / 255), (1 / gc));
		image[idx].x = r;
		image[idx].y = g;
		image[idx].z = b;
	}
}

GammaCorrectionImageCommand::GammaCorrectionImageCommand(float gc) : gc(gc) {}

void GammaCorrectionImageCommand::execute(uchar4** image, size_t* height, size_t* width) {
	uchar4* d_image;
	size_t in_h = *height;
	size_t in_w = *width;
	cudaMalloc(&d_image, in_h * in_w * sizeof(uchar4));
	cudaMemcpy(d_image, *image, in_h * in_w * sizeof(uchar4), cudaMemcpyHostToDevice);
	gammaCorrection <<< dim3(1 + ((in_h - 1) / 32), 1 + ((in_w - 1) / 32), 1), dim3(32, 32, 1) >>> (d_image, in_h, in_w, gc);
	cudaMemcpy(*image, d_image, in_h * in_w * sizeof(uchar4), cudaMemcpyDeviceToHost);
}

std::string GammaCorrectionImageCommand::toString() {
	return "GammaCorrection(" + std::to_string(gc) + ")";
}


__global__ void radial(uchar4* image_in, uchar4* image_out,
	size_t height, size_t width, float k1, float s) {
	int out_x = blockIdx.x * blockDim.x + threadIdx.x;
	int out_y = blockIdx.y * blockDim.y + threadIdx.y;
	float dx = out_x / (float)width - 0.5f;
	float dy = out_y / (float)height - 0.5f;
	float denom = s + k1 * (dx*dx + dy*dy);
	int in_x = (0.5f + dx / denom)*width;
	int in_y = (0.5f + dy / denom)*height;
	if (out_x < width && out_y < height) {
		if (0 <= in_x && in_x < width && 0 <= in_y && in_y < height) {
			image_out[width * out_y + out_x] = image_in[in_y * width + in_x];
		}
		/*else {
			image_out[width * out_y + out_x] = {0, 0, 0, 0};
		}*/
	}
}

RadialDistortionImageCommand::RadialDistortionImageCommand(float k1) : k1(k1) {}

void RadialDistortionImageCommand::execute(uchar4** image, size_t* height, size_t* width) {
	size_t in_h = *height;
	size_t in_w = *width;
	uchar4* d_in, * d_out;
	float scale;
	if (k1 > 0) scale = 1 - k1 / 2;
	else scale = 1 - k1 / 4;
	cudaMalloc(&d_in, in_h * in_w * sizeof(uchar4));
	cudaMalloc(&d_out, in_h * in_w * sizeof(uchar4));
	cudaMemcpy(d_in, *image, in_h * in_w * sizeof(uchar4), cudaMemcpyHostToDevice);
	radial <<< dim3(1 + ((in_w - 1) / 32), 1 + ((in_h - 1) / 32), 1), dim3(32, 32, 1) >>> (d_in, d_out, in_h, in_w, k1, scale);
	cudaMemcpy(*image, d_out, in_h * in_w * sizeof(uchar4), cudaMemcpyDeviceToHost);
}

std::string RadialDistortionImageCommand::toString() {
	return "RadialDistortion(" + std::to_string(k1) + ")";
}

__global__ void contrast(uchar4* image, size_t height, size_t width, float alpha) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < height && y < width) {
		int idx = x * width + y;
		image[idx].x = alpha * image[idx].x;
		image[idx].y = alpha * image[idx].y;
		image[idx].z = alpha * image[idx].z;
		if (image[idx].x > 255)
			image[idx].x = 255;
		if (image[idx].y > 255)
			image[idx].y = 255;
		if (image[idx].z > 255)
			image[idx].z = 255;
	}
}


void executeContrast(uchar4** image, size_t* height, size_t* width, float alpha) {
	size_t in_h = *height;
	size_t in_w = *width;
	uchar4* d_image;
	cudaMalloc(&d_image, in_h * in_w * sizeof(uchar4));
	cudaMemcpy(d_image, *image, in_h * in_w * sizeof(uchar4), cudaMemcpyHostToDevice);
	contrast <<< dim3(1 + ((in_h - 1) / 32), 1 + ((in_w - 1) / 32), 1), dim3(32, 32, 1) >>> (d_image, in_h, in_w, alpha);
	cudaMemcpy(*image, d_image, in_h * in_w * sizeof(uchar4), cudaMemcpyDeviceToHost);
}

__constant__ int mask[3 * 3];

__global__ void sharpen(uchar4* image_in, uchar4* image_out, size_t height, size_t width) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x<width && y<height) {
		int idx = y * width + x;

		int tempx = 0;
		int tempy = 0;
		int tempz = 0;
		for (int j = 0; j < 3; j++) {
			for (int i = 0; i < 3; i++) {
				int ypos = y - 1 + j;
				int xpos = x - 1 + i;
				int pos = ypos * width + xpos;
				bool nonBoundary = 0 <= ypos && ypos < height && 0 <= xpos && xpos < width;
				tempx += mask[j*3+i] * (nonBoundary ? image_in[pos].x : image_in[idx].x);
				tempy += mask[j*3+i] * (nonBoundary ? image_in[pos].y : image_in[idx].y);
				tempz += mask[j*3+i] * (nonBoundary ? image_in[pos].z : image_in[idx].z);
			}
		}
		tempx = tempx > 255 ? 255 : tempx;
		tempx = tempx < 0 ? 0 : tempx;
		tempy = tempy > 255 ? 255 : tempy;
		tempy = tempy < 0 ? 0 : tempy;
		tempz = tempz > 255 ? 255 : tempz;
		tempz = tempz < 0 ? 0 : tempz;
		image_out[idx].x = tempx;
		image_out[idx].y = tempy;
		image_out[idx].z = tempz;
	}
}

void SharpeningImageCommand::execute(uchar4** image, size_t* height, size_t* width) {
	size_t in_h = *height;
	size_t in_w = *width;
	uchar4 *d_in, *d_out;
	int filter[] = { 0,-1,0,-1,5,-1,0,-1,0 };
	cudaMalloc(&d_in, in_h * in_w * sizeof(uchar4));
	cudaMalloc(&d_out, in_h * in_w * sizeof(uchar4));
	cudaMemcpyToSymbol(mask, filter, 3*3*sizeof(int));
	cudaMemcpy(d_in, *image, in_h * in_w * sizeof(uchar4), cudaMemcpyHostToDevice);
	sharpen <<< dim3(1+((in_w-1)/32), 1+((in_h-1)/32), 1), dim3(32, 32, 1) >>> (d_in, d_out, in_h, in_w);
	cudaMemcpy(*image, d_out, in_h * in_w * sizeof(uchar4), cudaMemcpyDeviceToHost);
}

std::string SharpeningImageCommand::toString() {
	return "Sharpen()";
}
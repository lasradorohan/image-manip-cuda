#include "kernels.h"
#include "cmath"

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

void executeBlackWhite(uchar4** image, size_t *height, size_t *width) {
	size_t in_h = *height;
	size_t in_w = *width;
	uchar4* d_image;
	cudaMalloc(&d_image, in_h * in_w * sizeof(uchar4));
	cudaMemcpy(d_image, *image, in_h * in_w * sizeof(uchar4), cudaMemcpyHostToDevice);
	blackWhite<<<dim3(1 + ((in_h - 1) / 32), 1 + ((in_w - 1) / 32), 1), dim3(32, 32, 1)>>>(d_image, in_h, in_w);
	cudaMemcpy(*image, d_image, in_h * in_w * sizeof(uchar4), cudaMemcpyDeviceToHost);
	//image memory, height and width dont change
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

void executeRotate(uchar4** image, size_t *height, size_t *width, float phi) {
	size_t in_h = *height;
	size_t in_w = *width;
	size_t out_w = in_w * cos(phi) + in_h * sin(phi);
	size_t out_h = in_w * sin(phi) + in_h * cos(phi);
	uchar4 *image_out = new uchar4[out_w * out_h];
	uchar4 *d_in, *d_out;
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

__global__ void contrast(uchar4* image, size_t height, size_t width,float alpha) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < height && y < width) {
		int idx = x * width + y;
		image[idx].x = alpha * image[idx].x;
		image[idx].y = alpha * image[idx].y;
		image[idx].z = alpha * image[idx].z;
		if(image[idx].x > 255)
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
	contrast << <dim3(1 + ((in_h - 1) / 32), 1 + ((in_w - 1) / 32), 1), dim3(32, 32, 1) >> > (d_image, in_h, in_w,alpha);
	cudaMemcpy(*image, d_image, in_h * in_w * sizeof(uchar4), cudaMemcpyDeviceToHost);
}
#include <iostream>
#include <vector>
#include <sstream>

#include "InteractivePrompt.h"

auto testIndividualCommand() -> void {
	uchar4* image;
	size_t height, width;
	Image::loadImageRGBA(".\\resources\\4.jpg", &image, &height, &width);
	SkewImageCommand(0, 10).execute(&image, &height, &width);
	Image::saveImageRGBA(image, height, width, ".\\resources\\4_mod.jpg");
}

auto testCommandQueue() -> void {
	uchar4* image;
	size_t height, width;
	Image::loadImageRGBA(".\\resources\\testbw.jpg", &image, &height, &width);
	CommandQueue cq;
	cq.addCommand<BlackWhiteImageCommand>();
	cq.addCommand<RotateImageCommand>(0.349066f);
	cq.addCommand<RadialDistortionImageCommand>(-0.5f);
	cq.executeAll(&image, &height, &width);
	Image::saveImageRGBA(image, height, width, ".\\resources\\testbw_mod.jpg");
}

auto main() -> int {
	InteractivePrompt().promptLoop();
}
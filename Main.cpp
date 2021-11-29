#include <iostream>
#include <vector>
#include <sstream>

#include "InteractivePrompt.h"


auto main() -> int {
	InteractivePrompt prompt;
	prompt.promptLoop();
}

auto testIndividualCommand() -> void {
	uchar4* image;
	size_t height, width;
	Image::loadImageRGBA(".\\resources\\testbw.jpg", &image, &height, &width);
	SharpeningImageCommand().execute(&image, &height, &width);
	Image::saveImageRGBA(image, height, width, ".\\resources\\testbw_mod.jpg");
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
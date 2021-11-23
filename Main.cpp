#include <iostream>
#include <vector>
#include <sstream>

//#include "InteractivePrompt.h"

//#include "kernels.h"

#include "CommandQueue.h"


auto main() -> int {
	//RadialDistortionImageCommand(-0.5f).execute(&image, &height, &width);
	//InteractivePrompt prompt;
	//prompt.promptLoop();
	uchar4* image;
	size_t height, width;
	Dispatch::loadImageRGBA(".\\resources\\opera_house.jpg", &image, &height, &width);
	CommandQueue cq;
	cq.addCommand<BlackWhiteImageCommand>();
	cq.addCommand<RotateImageCommand>(0.349066f);
	cq.addCommand<RadialDistortionImageCommand>(-0.5f);
	cq.executeAll(&image, &height, &width);
	Dispatch::saveImageRGBA(image, height, width, ".\\resources\\opera_house_mod.jpg");
}
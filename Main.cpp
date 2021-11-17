#include <iostream>
#include <vector>
#include <sstream>

//#include "InteractivePrompt.h"

#include "kernels.h"

auto main() -> int
{
	//InteractivePrompt prompt;
	//prompt.promptLoop();
	uchar4* image;
	size_t height, width;
	Dispatch::loadImageRGBA(".\\resources\\opera_house.jpg", &image, &height, &width);
	executeBlackWhite(image, height, width);
	Dispatch::saveImageRGBA(image, height, width, ".\\resources\\opera_house_mod.jpg");
}
#include <iostream>
#include <vector>
#include <sstream>

//#include "InteractivePrompt.h"

#include "kernels.h"

#include "CommandQueue.h"

auto main() -> int
{
	//InteractivePrompt prompt;
	//prompt.promptLoop();
	uchar4* image;
	size_t height, width;
	Dispatch::loadImageRGBA(".\\resources\\opera_house.jpg", &image, &height, &width);
	float alpha = 1.15f;
	executeContrast(&image, &height, &width,alpha);
	Dispatch::saveImageRGBA(image, height, width, ".\\resources\\opera_house_mod.jpg");
	return 0;
}

//auto main() -> int {
//	CommandQueue cq;
//	cq.addCommand(std::make_shared<TempCommand>(123.0f));
//	cq.addCommand(std::make_shared<Temp2Command>(123.0f, 32.0f));
//	cq.printQueue();
//}
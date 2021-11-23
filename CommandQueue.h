#ifndef COMMAND_QUEUE
#define COMMAND_QUEUE

#include <list>
#include <memory>
#include <string>
#include "kernels.h"
//class Command {
//public:
//	virtual auto toString() -> const std::string = 0;
//};
//
//class TempCommand : public Command {
//	float radius;
//public:
//	TempCommand(float radius) : radius(radius) {
//	}
//	~TempCommand(){}
//	auto toString() -> const std::string {
//		return "TemporaryCommandv2(" + std::to_string(radius) + ")";
//	}
//};
//
//class Temp2Command : public Command {
//	float height, width;
//public:
//	Temp2Command(float height, float width) : height(height), width(width) {}
//	~Temp2Command() {}
//	auto toString() -> const std::string {
//		return "TemporaryCommandv2(" + std::to_string(height) + ", " + std::to_string(width) + ")";
//	}
//};

class CommandQueue {
	std::list<std::unique_ptr<ImageCommand>> queue;
public:
	/*auto addCommand(std::unique_ptr<ImageCommand> pCommand) -> void {
		queue.push_back(pCommand);
	}*/
	template <class T, class... _Types>
	auto addCommand(_Types&&... _Args) {
		queue.push_back(std::make_unique<T>(std::forward<_Types>(_Args)...));
		std::cout << "Command added, queue size=" << queue.size() << "\n";
	}
	auto removeCommand(int n = 1) -> void {
		if (n >= queue.size()) queue.clear();
		else queue.erase(std::prev(queue.end(), n), queue.end());
	}
	/*auto printQueue() -> void {
		for (const auto& pCommand : queue) {
			std::cout << pCommand->toString() << std::endl;
		}
	}*/
	auto executeAll(uchar4** image, size_t* height, size_t* width) -> void {
		for (auto&& command : queue) {
			command->execute(image, height, width);
		}
	}
	
};

#endif
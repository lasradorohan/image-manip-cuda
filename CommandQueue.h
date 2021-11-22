#ifndef COMMAND_QUEUE
#define COMMAND_QUEUE

#include <list>
#include <memory>
#include <string>

class Command {
public:
	virtual auto toString() -> const std::string = 0;
};

class TempCommand : public Command {
	float radius;
public:
	TempCommand(float radius) : radius(radius) {
	}
	~TempCommand(){}
	auto toString() -> const std::string {
		return "TemporaryCommandv2(" + std::to_string(radius) + ")";
	}
};

class Temp2Command : public Command {
	float height, width;
public:
	Temp2Command(float height, float width) : height(height), width(width) {}
	~Temp2Command() {}
	auto toString() -> const std::string {
		return "TemporaryCommandv2(" + std::to_string(height) + ", " + std::to_string(width) + ")";
	}
};

class CommandQueue {
	std::list<std::shared_ptr<Command>> queue;
public:
	auto addCommand(std::shared_ptr<Command> pCommand) -> void {
		queue.push_back(pCommand);
	}
	auto printQueue() -> void {
		for (const auto& pCommand : queue) {
			std::cout << pCommand->toString() << std::endl;
		}
	}
};

#endif
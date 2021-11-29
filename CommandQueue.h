#ifndef COMMAND_QUEUE_H_
#define COMMAND_QUEUE_H_

#include <list>
#include <memory>
#include <string>
#include "ImageCommands.h"

class CommandQueue {
	std::list<std::unique_ptr<ImageCommand>> queue;
public:
	template <class T, class... _Types>
	auto addCommand(_Types&&... _Args) {
		queue.push_back(std::make_unique<T>(std::forward<_Types>(_Args)...));
		//std::cout << "Command added, queue size=" << queue.size() << "\n";
	}
	auto clear(int n = -1) -> void {
		if (n==-1 || n >= queue.size()) queue.clear();
		else queue.erase(std::prev(queue.end(), n), queue.end());
	}
	auto printQueue(std::ostream& out) -> void {
		for (const auto& command : queue) {
			out << command->toString() << "\n";
		}
	}
	auto executeAll(uchar4** image, size_t* height, size_t* width) -> void {
		for (auto&& command : queue) {
			command->execute(image, height, width);
		}
	}
	
};

#endif

#ifndef INTERACTIVEPROMPT_H
#define INTERACTIVEPROMPT_H

#include <list>
#include <vector>
#include <iostream>
#include <string>

#include "Dispatch.h"

enum class Status{ CONTINUE, EXIT };

class InteractivePrompt {
public:
	Dispatch dispatch;
	static auto tokenize(const std::string& input)->std::list<std::string>;
	auto promptLoop() -> void;
	auto printPromptString() -> void;
	auto executeInput(std::list<std::string>& tokens) -> Status;
	auto invalidUsage(const std::string& command);
};


#endif
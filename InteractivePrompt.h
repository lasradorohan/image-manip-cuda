#ifndef INTERACTIVEPROMPT_H_
#define INTERACTIVEPROMPT_H_

#include <list>
#include <fstream>
#include <string>

#include "Dispatch.h"
#include "Status.h"


class InteractivePrompt {
public:
	Dispatch dispatch;
	static auto tokenize(const std::string& input)->std::list<std::string>;
	auto promptLoop() -> void;
	auto printPromptString() -> void;
	auto executeInput(std::list<std::string>& tokens) -> Status;
	auto invalidUsage(const std::string& command);
	auto displayHelp(std::string command) -> void;
	auto displayGeneralHelp() -> void;

};


#endif
#include "InteractivePrompt.h"

auto InteractivePrompt::tokenize(const std::string& input) -> std::list<std::string> {
    std::list<std::string> tokens;
    std::string temp;
    char delim = ' ';
    size_t current = 0, next;
    while(true) {
        while (input.at(current) == ' ') current++;
        if (input.at(current) == '\"') {
            current++;
            delim = '\"';
        } else {
            delim = ' ';
        }
        next = input.find(delim, current);
        if (next == std::string::npos) next = input.size();
        tokens.push_back(input.substr(current, next-current));
        current = next+1;
        if (current >= input.size()) break;
    }
    return tokens;
}

auto InteractivePrompt::invalidUsage(const std::string& command) {
    std::cout << "Invalid usage of `" << command << "`\n"
        << "Type help " << command << " for more info.\n";
}

auto InteractivePrompt::executeInput(std::list<std::string>& tokens) -> Status {
    std::string& command = *tokens.begin();
    Status ret(Status::CONTINUE);
    if (command.compare("load") == 0) {
        if (tokens.size() < 2) {
            invalidUsage(command);
        }
        dispatch.clearFiles();
        auto it = tokens.begin();
        it++;
        for (; it != tokens.end(); it++) {
            auto err = dispatch.addFiles(*it);
            if (err == ErrorCode::FileNotFound) {
                std::cout << "No images found corresponding to \"" << *it << "\"\n";
            } else if(err == ErrorCode::NotImage){
                std::cout << *it << " is not an image or directory";
            }
            
        }
        std::cout << "Loaded " << dispatch.numFiles() << " images.\n";
        
    }
    else if (command.compare("listqueue") == 0) {
    }
    else if (command.compare("clear") == 0) {
    }
    else if (command.compare("process") == 0) {
    }
    else if (command.compare("save") == 0) {
    }
    else if (command.compare("view") == 0) {
    }
    else if (command.compare("listfiles") == 0) {
    }
    else if (command.compare("exit") == 0) {
        ret = Status::EXIT;
    }
    else {
        // TODO: other commands
        
    }
    return ret;
}

auto InteractivePrompt::printPromptString() -> void {
    std::cout << "> ";
}

auto InteractivePrompt::promptLoop() -> void {
    while (true) {
        printPromptString();
        std::string input;
        std::getline(std::cin, input);
        std::list<std::string> tokens = tokenize(input);
        Status status = executeInput(tokens);
        if (status == Status::EXIT) return;
    }
}

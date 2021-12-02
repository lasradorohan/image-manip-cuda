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
    Status ret(Status::Success);
    if (command.compare("load") == 0) {
        if (tokens.size() < 2) return Status::InvalidUsage;
        dispatch.clearFiles();
        auto it = tokens.begin();
        it++;
        for (; it != tokens.end(); it++) {
            auto err = dispatch.imageQueue.addFiles(*it);
            if (err == Status::FileNotFound) {
                std::cout << "No images found corresponding to \"" << *it << "\"\n";
            } else if(err == Status::NotImage){
                std::cout << *it << " is not an image or directory\n";
            }
            
        }
        std::cout << "Loaded " << dispatch.numFiles() << " images.\n";
        dispatch.imageQueue.loadAll();
        
    }
    else if (command.compare("listqueue") == 0) {
        dispatch.commandQueue.printQueue(std::cout);
    }
    else if (command.compare("clear") == 0) {
        if (tokens.size() == 1) dispatch.commandQueue.clear();
        else  {
            int n = std::stoi(*++tokens.begin());
            if (n == 0 || tokens.size() > 2) {
                return Status::InvalidUsage;
            }
            dispatch.commandQueue.clear(n);
        }
    }
    else if (command.compare("process") == 0) {
        if (tokens.size() > 1) {
            //eager execution
            tokens.pop_front();
            dispatch.commandQueue.clear();      //TODO: fix this side effect
            Status status = executeInput(tokens);
            if(status!=Status::Success) return Status::InvalidUsage;
            dispatch.process();
        }
        else {
            dispatch.process();
        }
    }
    else if (command.compare("save") == 0) {
        if (tokens.size() != 2) return Status::InvalidUsage;
        auto param = std::next(tokens.begin(), 1);
        if (param->compare("overwrite") == 0) dispatch.saveOverwrite();
        else dispatch.saveTo(*param);
    }
    else if (command.compare("view") == 0) {
        if (tokens.size() > 2) return Status::InvalidUsage;
        int idx = 0;
        if (tokens.size() == 2) idx = std::stoi(*std::next(tokens.begin(), 1));
        dispatch.imageQueue.display(idx);
    }
    else if (command.compare("listfiles") == 0) {
        dispatch.imageQueue.listFiles(std::cout);
    }
    else if (command.compare("exit") == 0) {
        ret = Status::Exit;
    }
    else if(command.compare("bw") == 0) {
        if (tokens.size() != 1) return Status::InvalidUsage;
        dispatch.commandQueue.addCommand<BlackWhiteImageCommand>();
    }
    else if (command.compare("rotate") == 0) {
        if (tokens.size() != 2) return Status::InvalidUsage;
        float param = std::stof(*std::next(tokens.begin(), 1));
        dispatch.commandQueue.addCommand<RotateImageCommand>(param);
    }
    else if (command.compare("gamma") == 0) {
        if (tokens.size() != 2) return Status::InvalidUsage;
        float param = std::stof(*std::next(tokens.begin(), 1));
        dispatch.commandQueue.addCommand<GammaCorrectionImageCommand>(param);
    }
    else if (command.compare("radial") == 0) {
        if (tokens.size() != 2) return Status::InvalidUsage;
        float param = std::stof(*std::next(tokens.begin(), 1));
        dispatch.commandQueue.addCommand<RadialDistortionImageCommand>(param);
    }
    else if (command.compare("contrast") == 0) {
        if (tokens.size() != 2) return Status::InvalidUsage;
        float param = std::stof(*std::next(tokens.begin(), 1));
       dispatch.commandQueue.addCommand<ContrastImageCommand>(param);
    }
    else if (command.compare("sharpen") == 0) {
        if (tokens.size() > 1) return Status::InvalidUsage;
        dispatch.commandQueue.addCommand<SharpeningImageCommand>();
    }
    else if (command.compare("skew") == 0) {
        if (tokens.size() != 3) return Status::InvalidUsage;
        auto it = tokens.begin();
        float param1 = std::stof(*(++it));
        float param2 = std::stof(*(++it));
        dispatch.commandQueue.addCommand<SkewImageCommand>(param1, param2);
    }
    else if (command.compare("gaussianblur") == 0) {
        if (tokens.size() > 1) return Status::InvalidUsage;
        dispatch.commandQueue.addCommand<GaussianBlurImageCommand>();
    }
    else if (command.compare("help") == 0) {
        if (tokens.size() == 2) {
            displayHelp(*(++tokens.begin()));
        }
        else if (tokens.size() == 1) {
            displayGeneralHelp();
        }
        else {
            return Status::InvalidUsage;
        }
    }
    // TODO: other commands
    else {

    }
    return ret;
}

auto InteractivePrompt::displayHelp(std::string command) -> void {
    std::ifstream in;
    /*if (command.size() == 0) {
        in.open("./help/help.txt");
    }
    else */if (std::filesystem::is_regular_file(".\\help\\help_" + command + ".txt")) {
        in.open(".\\help\\help_" + command + ".txt");
    }
    else {
        std::cout << "No help page found for `" << command << "`" << std::endl;
        return;
    }
    std::cout << std::endl << std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()) << std::endl << std::endl;
}

auto InteractivePrompt::displayGeneralHelp() -> void {
    std::ifstream in("./help/help.txt");
    std::cout << std::endl << std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()) << std::endl << std::endl;

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
        std::string currentToken = *tokens.begin();
        Status status = executeInput(tokens);
        if (status == Status::InvalidUsage) invalidUsage(currentToken);
        if (status == Status::Exit) return;
    }
}

#ifndef DISPATCH_H_
#define DISPATCH_H_


#include <filesystem>
#include "cuda_runtime.h"
#include "CommandQueue.h"
#include "Status.h"
#include "ImageQueue.h"
#include "ProgressBar.h"


class Dispatch {
public:
    CommandQueue commandQueue;
    ImageQueue imageQueue;
   
    auto clearFiles() { imageQueue.clear(); }

    auto numFiles() { return imageQueue.size(); }

    auto process() -> void {
        uchar4* imagePtr;
        size_t height, width;
        ProgressBar progress(imageQueue.size());

        for (auto&& image : imageQueue) {
            if (!image.isLoaded) image.load();
            image.loadRGBA(&imagePtr, &height, &width);
            commandQueue.executeAll(&imagePtr, &height, &width);
            image.saveRGBATemp(imagePtr, height, width);
            delete[] imagePtr;
            progress.step();
        }
        std::cout << std::endl;
    }

    auto saveOverwrite() -> void {
        for (auto&& image : imageQueue) {
            image.save();
        }
    }

    auto saveTo(const std::string& dir) {
        for (auto&& image : imageQueue) {
            image.saveTo(std::filesystem::path(dir));
        }
    }
};

#endif
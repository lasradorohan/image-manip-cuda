#ifndef DISPATCH
#define DISPATCH

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <stdio.h>
#include "cuda_runtime.h"
#include <filesystem>

enum class ErrorCode {Success, FileNotFound, NotImage, Error};

class Dispatch {

private:
    std::list<std::filesystem::path> files;
public:
    static auto loadImageRGBA(const std::string& filename,
        uchar4** imagePtr,
        size_t* numRows, size_t* numCols)
    {
        cv::Mat image = cv::imread(filename.c_str(), cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Couldn't open file: " << filename << std::endl;
            exit(1);
        }

        if (image.channels() != 3) {
            std::cerr << "Image must be color!" << std::endl;
            exit(1);
        }

        if (!image.isContinuous()) {
            std::cerr << "Image isn't continuous!" << std::endl;
            exit(1);
        }

        cv::Mat imageRGBA;
        cv::cvtColor(image, imageRGBA, cv::COLOR_BGR2RGBA);

        *imagePtr = new uchar4[image.rows * image.cols];

        unsigned char* cvPtr = imageRGBA.ptr<unsigned char>(0);
        for (size_t i = 0; i < image.rows * image.cols; ++i) {
            (*imagePtr)[i].x = cvPtr[4 * i + 0];
            (*imagePtr)[i].y = cvPtr[4 * i + 1];
            (*imagePtr)[i].z = cvPtr[4 * i + 2];
            (*imagePtr)[i].w = cvPtr[4 * i + 3];
        }

        *numRows = image.rows;
        *numCols = image.cols;
    }

    static auto saveImageRGBA(const uchar4* const image,
        const size_t numRows, const size_t numCols,
        const std::string& output_file)
    {
        int sizes[2];
        sizes[0] = numRows;
        sizes[1] = numCols;
        cv::Mat imageRGBA(2, sizes, CV_8UC4, (void*)image);
        cv::Mat imageOutputBGR;
        cv::cvtColor(imageRGBA, imageOutputBGR, cv::COLOR_RGBA2BGR);
        //output the image
        cv::imwrite(output_file.c_str(), imageOutputBGR);
    }

    auto isImage(std::filesystem::path p) -> bool {
        std::string ext = p.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext.compare("png") == 0 || ext.compare("jpg") == 0 || ext.compare("jpeg") == 0 || ext.compare("gif") == 0 || ext.compare("tiff") == 0) {
            return true;
        }
        return false;
    }

    auto clearFiles() { this->files.clear(); }

    auto addFiles(std::string path) -> ErrorCode {
        std::filesystem::path p = path;
        if (std::filesystem::is_regular_file(p)) {
            if (isImage(p)) {
                this->files = { p };
                return ErrorCode::Success;
            }
            return ErrorCode::NotImage;
        } else if (std::filesystem::is_directory(p)) {
            for (std::filesystem::directory_iterator iter(p); iter != std::filesystem::directory_iterator(); iter++) {
                if (std::filesystem::is_regular_file(*iter) && isImage(*iter)) {
                    this->files.push_back(*iter);
                }
            }
        }
        if (this->files.size() == 0) return ErrorCode::FileNotFound;
        return ErrorCode::Success;
    }
    auto numFiles() { return this->files.size(); }
};

#endif
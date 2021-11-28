#ifndef IMAGE_QUEUE_H_
#define IMAGE_QUEUE_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <unordered_map>
#include <sstream>
#include "cuda_runtime.h"
#include "Status.h"


namespace fs = std::filesystem;

const fs::path TEMP_DIR = fs::absolute(".\\.temp\\");

class Image {
    std::hash<std::string> hasher;
public:
	fs::path originPath;
    fs::path tempPath;
    bool isLoaded = false;
    

    Image(fs::path path) : originPath(path) {
        std::ostringstream oss;
        oss << std::hex << hasher(path.string()) << path.extension().string();
        tempPath = TEMP_DIR / oss.str();
    }

    auto currentPath() const -> const fs::path& {
        if (isLoaded) return tempPath;
        else return originPath;
    }

    void load() {
        fs::copy_file(originPath, tempPath, fs::copy_options::overwrite_existing);
        isLoaded = true;
    }
    void save() {
        fs::copy_file(tempPath, originPath, fs::copy_options::overwrite_existing);
    }
    void saveTo(const fs::path& path) {
        fs::copy_file(currentPath(), path / originPath.filename(), fs::copy_options::overwrite_existing);
    }

    static auto loadImageRGBA(const std::string& filename, uchar4** imagePtr, size_t* numRows, size_t* numCols) {
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

    auto loadRGBA(uchar4** imagePtr, size_t* numRows, size_t* numCols) {
        loadImageRGBA(currentPath().string(), imagePtr, numRows, numCols);
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
        cv::imwrite(output_file.c_str(), imageOutputBGR);
    }

    auto saveRGBATemp(const uchar4* const image, const size_t numRows, const size_t numCols) {
        saveImageRGBA(image, numRows, numCols, tempPath.string());
    }

    static auto isImage(const fs::path& p) -> bool {
        std::string ext = p.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext.compare(".png") == 0 || ext.compare(".jpg") == 0 || ext.compare(".jpeg") == 0 || ext.compare(".gif") == 0 || ext.compare(".tiff") == 0) {
            return true;
        }
        return false;
    }

    auto display() {
        cv::Mat image = cv::imread(currentPath().string(), cv::IMREAD_COLOR);
        cv::imshow(originPath.filename().string(), image);
    }
};

class ImageQueue {
    std::list<Image> queue;
public:
    size_t size() {
        return queue.size();
    }
    void loadAll() {
        for (auto&& image : queue) {
            image.load();
        }
    }
    void addImage(fs::path p) {
        queue.push_back(Image(p));
    }
    void clear() {
        queue.clear();
    }

    auto addFiles(std::string path) -> Status {
        fs::path p = path;
        if (fs::is_regular_file(p)) {
            if (Image::isImage(p)) {
                queue.clear();
                addImage(p);
                return Status::Success;
            }
            return Status::NotImage;
        }
        else if (fs::is_directory(p)) {
            for (fs::directory_iterator iter(p); iter != fs::directory_iterator(); iter++) {
                if (fs::is_regular_file(*iter) && Image::isImage(*iter)) {
                    addImage(*iter);
                }
            }
        }
        if (size() == 0) return Status::FileNotFound;
        return Status::Success;
    }

    auto display(size_t idx = 0) {
        if(idx<queue.size()) std::next(queue.begin(), idx)->display();
    }

    typedef typename std::list<Image>::iterator iterator;
    typedef typename std::list<Image>::const_iterator const_iterator;

    iterator begin() { return queue.begin(); }
    const_iterator begin() const { return queue.begin(); }
    const_iterator cbegin() const { return queue.cbegin(); }
    iterator end() { return queue.end(); }
    const_iterator end() const { return queue.end(); }
    const_iterator cend() const { return queue.cend(); }

    auto listFiles(std::ostream& out) {
        int count = 0;
        for (const auto& image : queue) {
            out << count++ << ": " << image.originPath.string() << std::endl;
        }
    }
};

#endif
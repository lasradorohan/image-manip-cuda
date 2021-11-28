#ifndef PROGRESS_BAR_H_
#define PROGRESS_BAR_H_

#include <iostream>

class ProgressBar {
    int length, charCount, val;
public:
    ProgressBar(int length, int val = 0, int charCount = 70) : length(length), charCount(charCount), val(val) {
        setProgress(val);
    }
    /* ProgressBar(int start, int end, int charCount = 70): start(start), length(end-start), charCount(charCount) {
         setProgress(0);
     }*/
    void setProgress(int val) {
        std::cout << "[";
        float progress = val / (float)length;
        int pos = progress * charCount;
        for (int i = 0; i < charCount; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.f) << " %\r";
        std::cout.flush();
    }
    void step() {
        setProgress(++val);
    }
};

#endif
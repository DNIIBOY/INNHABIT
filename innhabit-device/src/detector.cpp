#include "detector.h"
#include "common.h"
#include <algorithm>
#include <chrono>

// Implement the constructor (removed initialize call)
GenericDetector::GenericDetector(const string& model_path, const vector<string>& target_classes)
    : target_classes_(target_classes) {
    // Derived classes must call initialize(model_path) explicitly!
}

// Implement detect function
void GenericDetector::detect(cv::Mat& frame) {
    if (!initialized_) {
        std::cerr << "Detector not properly initialized" << std::endl;
        return;
    }
    detections_.clear();

    preprocess(frame);

    inference(frame);

    postprocess(frame);
}

// Implement helper function to clamp values
int GenericDetector::clamp(int val, int min_val, int max_val) {
    return std::max(min_val, std::min(val, max_val));
}

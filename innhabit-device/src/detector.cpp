#include "detector.h"
#include "common.h"
#include <algorithm>

// Default constructor for the GenericDetector
GenericDetector::GenericDetector(const string& model_path, const vector<string>& target_classes)
    : target_classes_(target_classes), initialized_(false) {
    // Implementation moved to initialize() which is called by derived classes
}

// Main detection function - common for all platforms
void GenericDetector::detect(cv::Mat& frame) {
    if (!initialized_) {
        ERROR("Detector not properly initialized");
        return;
    }

    // Common preprocessing
    preprocess(frame);
    inference(frame);
    postprocess(frame);

}

// Helper function to clamp values
int GenericDetector::clamp(int val, int min_val, int max_val) {
    return max(min_val, min(val, max_val));
}

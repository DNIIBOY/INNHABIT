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
    frame_count_++;
    detections_.clear();

    auto startTime = std::chrono::high_resolution_clock::now();
    preprocess(frame);
    auto endTime = std::chrono::high_resolution_clock::now();
    float preprocess_time = std::chrono::duration<float, std::milli>(endTime - startTime).count();
    preprocess_times_.push_back(preprocess_time);

    startTime = std::chrono::high_resolution_clock::now();
    inference(frame);
    endTime = std::chrono::high_resolution_clock::now();
    float inference_time = std::chrono::duration<float, std::milli>(endTime - startTime).count();
    inference_times_.push_back(inference_time);

    startTime = std::chrono::high_resolution_clock::now();
    postprocess(frame);
    endTime = std::chrono::high_resolution_clock::now();
    float postprocess_time = std::chrono::duration<float, std::milli>(endTime - startTime).count();
    postprocess_times_.push_back(postprocess_time);

    if (frame_count_ >= 100) {
        float avg_preprocess = std::accumulate(preprocess_times_.begin(), 
                                             preprocess_times_.end(), 0.0f) / 100.0f;
        float avg_inference = std::accumulate(inference_times_.begin(), 
                                            inference_times_.end(), 0.0f) / 100.0f;
        float avg_postprocess = std::accumulate(postprocess_times_.begin(), 
                                              postprocess_times_.end(), 0.0f) / 100.0f;

        cout << "Average timing over 100 frames:" << std::endl;
        cout << "  Preprocess: " << avg_preprocess << " ms" << std::endl;
        cout << "  Inference: " << avg_inference << " ms" << std::endl;
        cout << "  Postprocess: " << avg_postprocess << " ms" << std::endl;
        cout << "  Total: " << (avg_preprocess + avg_inference + avg_postprocess) 
                 << " ms" << std::endl;

        preprocess_times_.clear();
        inference_times_.clear();
        postprocess_times_.clear();
        frame_count_ = 0;
    }
}

// Implement helper function to clamp values
int GenericDetector::clamp(int val, int min_val, int max_val) {
    return std::max(min_val, std::min(val, max_val));
}

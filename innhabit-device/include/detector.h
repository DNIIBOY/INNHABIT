#ifndef DETECTOR_H
#define DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

/**
    * Structure used for storing detection results
*/
struct Detection {
    string class_id;  
    float confidence;
    Rect bounding_box;
};

/**
    * Structure used for storing platform-specific inference outputs
*/
struct DetectionOutput {
    vector<void*> buffers;   // Pointers to output buffers
    vector<Mat> outputs;     // For OpenCV DNN outputs
    
    // Clear all data
    void clear() const {
        const_cast<vector<void*>&>(buffers).clear();
        const_cast<vector<Mat>&>(outputs).clear();
    }
};

/**
    * Generic detector implementation with common functionality
    * Able to be used on most platform-specific
*/
class GenericDetector {
protected:
    vector<string> target_classes_;
    bool initialized_ = false;
    vector<Detection> detections_;

    // Virtual methods to be implemented by platform-specific detectors
    virtual void initialize(const string& model_path) = 0;
    virtual void releaseOutputs() = 0;
    int clamp(int value, int min_value, int max_value);
    virtual void preprocess(cv::Mat& frame) = 0;
    virtual void inference(cv::Mat& frame) = 0;
    virtual void postprocess(cv::Mat& frame) = 0;

public:
    GenericDetector(const string& model_path, const vector<string>& target_classes);
    virtual ~GenericDetector() = default;
    virtual void detect(cv::Mat& frame);
    const vector<Detection>& getDetections() const { return detections_; }
    bool isInitialized() const { return initialized_; }
};
/** 
    * Factory function to create the appropriate detector
*/
GenericDetector* createDetector(const string& model_path, const vector<string>& target_classes);

#endif // DETECTOR_H

#ifndef DETECTOR_H
#define DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

// Structure for storing detection results
struct Detection {
    string class_id;  // Renamed from classId
    float confidence;
    Rect bounding_box; // Renamed from box
};

// Structure for storing platform-specific inference outputs
struct DetectionOutput {
    vector<void*> buffers;   // Pointers to output buffers
    vector<Mat> outputs;     // For OpenCV DNN outputs
    // Clear all data
    void clear() const {
        const_cast<vector<void*>&>(buffers).clear();
        const_cast<vector<Mat>&>(outputs).clear();
    }
};

// Abstract base class for all detectors
class Detector {
protected:
    vector<Detection> detections_;
public:
    virtual ~Detector() = default;
    virtual void detect(Mat& frame) = 0;
    virtual const vector<Detection>& getDetections() const { return detections_; }
};

// Generic detector implementation with common functionality
class GenericDetector : public Detector {
protected:
    vector<string> target_classes_;  // Renamed from targetClasses
    bool initialized_;
    vector<Detection> detections_;

    // Virtual methods to be implemented by platform-specific detectors
    virtual void initialize(const string& model_path) = 0;
    virtual void releaseOutputs(const DetectionOutput& output) = 0;
    int clamp(int value, int min_value, int max_value);  // Renamed parameters for clarity
    virtual void preprocess(cv::Mat& frame) = 0;
    virtual void inference(cv::Mat& frame) = 0;
    virtual void postprocess(cv::Mat& frame) = 0;

public:
    GenericDetector(const string& model_path, const vector<string>& target_classes);
    virtual ~GenericDetector() = default;
    void detect(Mat& frame) override;
    const vector<Detection>& getDetections() const { return detections_; }
    bool isInitialized() const { return initialized_; }
};

// Factory function to create the appropriate detector
Detector* createDetector(const string& model_path, const vector<string>& target_classes);

#endif // DETECTOR_H

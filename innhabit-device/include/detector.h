#ifndef DETECTOR_H
#define DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

// Structure for storing detection results
struct Detection {
    string classId;
    float confidence;
    Rect box;
};

// Structure for storing platform-specific inference outputs
struct DetectionOutput {
    vector<void*> buffers;   // Pointers to output buffers
    vector<float> scales;    // Scale factors for quantized outputs
    vector<int32_t> zps;     // Zero points for quantized outputs
    int num_outputs;         // Number of output tensors
    vector<Mat> outputs;     // For OpenCV DNN outputs

    // Default constructor
    DetectionOutput() : num_outputs(0) {}

    // Clear all data
    void clear() const {
        // Clear all vectors
        const_cast<vector<void*>&>(buffers).clear();
        const_cast<vector<float>&>(scales).clear();
        const_cast<vector<int32_t>&>(zps).clear();
        const_cast<int&>(num_outputs) = 0;
        const_cast<vector<Mat>&>(outputs).clear();
    }
};

// Abstract base class for all detectors
class Detector {
public:
    virtual ~Detector() = default;
    virtual void detect(Mat& frame) = 0;
    virtual const vector<Detection>& getDetections() const = 0;
};

// Generic detector implementation with common functionality
class GenericDetector : public Detector {
protected:
    vector<string> targetClasses;
    bool initialized;
    int width, height, channel;
    vector<Detection> detections;

    // Virtual methods to be implemented by platform-specific detectors
    virtual void initialize(const string& modelPath) = 0;
    virtual DetectionOutput runInference(const Mat& input) = 0;
    virtual void releaseOutputs(const DetectionOutput& output) = 0;
    virtual void processDetections(const DetectionOutput& output, Mat& frame, float scale, int dx, int dy);

public:
    GenericDetector(const string& modelPath, const vector<string>& targetClasses_);
    virtual ~GenericDetector() = default;

    void detect(Mat& frame) override;
    const vector<Detection>& getDetections() const override;
    int clamp(int val, int min_val, int max_val);
    bool isInitialized() const { return initialized; } // New public getter
};

// Factory function to create the appropriate detector
Detector* createDetector(const string& modelPath, const vector<string>& targetClasses);

// Utility functions declarations
Mat preprocessImage(const Mat& frame, int targetWidth, int targetHeight, float& scale, int& dx, int& dy);
void drawDetections(Mat& frame, const vector<Detection>& detections);

#endif // DETECTOR_H
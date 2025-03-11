#ifndef DETECTOR_H
#define DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// Forward declaration
struct Detection;

struct DetectionOutput {
    std::vector<void*> buffers;
    std::vector<float> scales;
    std::vector<int32_t> zps;
    int num_outputs;
};

// Structure to hold detection information
struct Detection {
    std::string classId;
    float confidence;
    cv::Rect box;
};

class Detector {
public:
    virtual ~Detector() {}
    virtual void detect(cv::Mat& frame) = 0;
    virtual const std::vector<Detection>& getDetections() const = 0;
protected:
    std::vector<std::string> targetClasses;
    int width, height, channel;
    bool initialized;
};

class GenericDetector : public Detector {
public:
    GenericDetector(const std::string& modelPath, const std::vector<std::string>& targetClasses_);
    void detect(cv::Mat& frame) override;
    const std::vector<Detection>& getDetections() const override;

protected:
    std::vector<Detection> detections;
    virtual void initialize(const std::string& modelPath) = 0;
    virtual DetectionOutput runInference(const cv::Mat& input) = 0;
    virtual void releaseOutputs(const DetectionOutput& output) {}  // Optional for platforms needing cleanup
};

Detector* createDetector(const std::string& modelPath, const std::vector<std::string>& targetClasses);

#endif // DETECTOR_H
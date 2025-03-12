#ifndef DETECTOR_H
#define DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

struct Detection {
    string classId;
    float confidence;
    Rect box;
};

struct DetectionOutput {
    vector<void*> buffers;
    vector<float> scales;
    vector<int32_t> zps;
    int num_outputs;
};

class Detector {
public:
    virtual ~Detector() {}
    virtual void detect(Mat& frame) = 0;
    virtual const vector<Detection>& getDetections() const = 0;

protected:
    virtual void initialize(const string& modelPath) = 0;
    virtual DetectionOutput runInference(const Mat& input) = 0;
    virtual void releaseOutputs(const DetectionOutput& output) = 0;
};

class GenericDetector : public Detector {
protected:
    vector<string> targetClasses;
    bool initialized;
    int width;
    int height;
    int channel;
    vector<Detection> detections;

public:
    GenericDetector(const string& modelPath, const vector<string>& targetClasses_);
    void detect(Mat& frame) override;
    const vector<Detection>& getDetections() const override;
    int clamp(int val, int min_val, int max_val); // Add clamp declaration

protected:
    void initialize(const string& modelPath) override = 0;
    virtual DetectionOutput runInference(const Mat& input) override = 0;
    virtual void releaseOutputs(const DetectionOutput& output) override = 0;
};

extern "C" Detector* createDetector(const string& modelPath, const vector<string>& targetClasses);

#endif // DETECTOR_H
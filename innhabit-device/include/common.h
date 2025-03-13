#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "detector.h"  // Include detector.h for Detection struct

using namespace std;
using namespace cv;

// Detection threshold constants (configurable via defines or runtime if needed)
#ifndef BOX_THRESH
#define BOX_THRESH 0.25f
#endif
#ifndef NMS_THRESH
#define NMS_THRESH 0.45f
#endif

// Default image dimensions (can be overridden by specific detectors)
#ifndef DETECTOR_WIDTH
#define DETECTOR_WIDTH 640
#endif
#ifndef DETECTOR_HEIGHT
#define DETECTOR_HEIGHT 640
#endif
#ifndef DETECTOR_CHANNELS
#define DETECTOR_CHANNELS 3
#endif

// COCO class labels (80 classes)
static const char* const COCO_LABELS[80] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

// Utility function declarations
void printUsage(const string& programName);
Mat preprocessImage(const Mat& frame, int targetWidth, int targetHeight, float& scale, int& dx, int& dy);
void drawDetections(Mat& frame, const vector<Detection>& detections);

// Logging helpers
#define LOG(msg) cout << "[INFO] " << msg << endl
#define ERROR(msg) cerr << "[ERROR] " << msg << endl

#endif // COMMON_H

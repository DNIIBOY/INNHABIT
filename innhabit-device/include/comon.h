#ifndef COMMON_H
#define COMMON_H

// Standard libraries
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

// Namespace shortcuts
using namespace std;
using namespace cv;

// Detection threshold constants
const float BOX_THRESH = 0.5;      // Minimum confidence for a valid box
const float NMS_THRESH = 0.45;     // Non-Maximum Suppression threshold

// Image dimensions (for model input resizing)
const int width = 640;
const int height = 640;

// Structure for detection result
struct Detection {
    string classId;      // Detected class label
    float confidence;    // Detection confidence score
    Rect box;            // Bounding box (x, y, width, height)
};

// Utility functions (if needed)
void printUsage(const string& programName) {
    cerr << "Usage: " << programName << " [--video <video_file>] [--image <image_file>] [--model <model_path>]" << endl;
}

// Logging helper (optional for debugging)
#define LOG(msg) cout << "[INFO] " << msg << endl;
#define ERROR(msg) cerr << "[ERROR] " << msg << endl;

#endif // COMMON_H

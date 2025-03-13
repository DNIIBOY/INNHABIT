#include "detector.h"
#include "common.h"
#include <algorithm>

// Default constructor for the GenericDetector
GenericDetector::GenericDetector(const string& modelPath, const vector<string>& targetClasses_)
    : targetClasses(targetClasses_), initialized(false), width(DETECTOR_WIDTH), height(DETECTOR_HEIGHT), channel(DETECTOR_CHANNELS) {
    // Implementation moved to initialize() which is called by derived classes
}

// Main detection function - common for all platforms
void GenericDetector::detect(Mat& frame) {
    if (!initialized) {
        ERROR("Detector not properly initialized");
        return;
    }

    // Common preprocessing
    float scale;
    int dx, dy;
    Mat resized_img = preprocessImage(frame, width, height, scale, dx, dy);
    
    // Platform-specific inference (implemented by derived classes)
    DetectionOutput output = runInference(resized_img);
    
    // Process detections (each platform will implement this)
    processDetections(output, frame, scale, dx, dy);
    
    // Draw detection boxes (common for all platforms)
    drawDetections(frame, detections);
    
    // Cleanup (platform-specific)
    releaseOutputs(output);
}

// Return the current detections
const vector<Detection>& GenericDetector::getDetections() const {
    return detections;
}

// Helper function to clamp values
int GenericDetector::clamp(int val, int min_val, int max_val) {
    return max(min_val, min(val, max_val));
}

// Common preprocessing function - moved from detector.cpp to implementation
Mat preprocessImage(const Mat& frame, int targetWidth, int targetHeight, float& scale, int& dx, int& dy) {
    Mat img;
    cvtColor(frame, img, COLOR_BGR2RGB);
    int img_width = img.cols;
    int img_height = img.rows;
    
    Mat resized_img(targetHeight, targetWidth, CV_8UC3, Scalar(114, 114, 114));
    scale = min(static_cast<float>(targetWidth) / img_width, static_cast<float>(targetHeight) / img_height);
    int new_width = static_cast<int>(img_width * scale);
    int new_height = static_cast<int>(img_height * scale);
    dx = (targetWidth - new_width) / 2;
    dy = (targetHeight - new_height) / 2;

    Mat resized_part;
    resize(img, resized_part, Size(new_width, new_height));
    resized_part.copyTo(resized_img(Rect(dx, dy, new_width, new_height)));
    
    return resized_img;
}

// Common drawing function - moved from detector.cpp to implementation
void drawDetections(Mat& frame, const vector<Detection>& detections) {
    for (const auto& det : detections) {
        rectangle(frame, det.box, Scalar(0, 255, 0), 2);

        char text[256];
        sprintf(text, "%s %.1f%%", det.classId.c_str(), det.confidence * 100);
        int baseLine = 0;
        Size label_size = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        
        int y = det.box.y - label_size.height - baseLine;
        if (y < 0) y = det.box.y + label_size.height;
        
        int x = det.box.x;
        if (x + label_size.width > frame.cols)
            x = frame.cols - label_size.width;
        
        rectangle(frame, Point(x, y - label_size.height - baseLine),
                  Point(x + label_size.width, y + baseLine),
                  Scalar(255, 255, 255), -1);
        putText(frame, text, Point(x, y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    }
}

// Generic implementation of processDetections that can be overridden by platform-specific code
void GenericDetector::processDetections(const DetectionOutput& output, Mat& frame, float scale, int dx, int dy) {
    // Default implementation does nothing
    // Platform-specific implementations should override this
}
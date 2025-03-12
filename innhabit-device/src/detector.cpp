#include "detector.h"
#include "postprocess.h"
#include "common.h"
#include <algorithm>

GenericDetector::GenericDetector(const string& modelPath, const vector<string>& targetClasses_)
    : targetClasses(targetClasses_), initialized(false), width(DETECTOR_WIDTH), height(DETECTOR_HEIGHT), channel(DETECTOR_CHANNELS) {
}

void GenericDetector::detect(Mat& frame) {
    if (!initialized) {
        ERROR("Detector not properly initialized");
        return;
    }

    float scale;
    int dx, dy;
    Mat resized_img = preprocessImage(frame, width, height, scale, dx, dy);
    DetectionOutput output = runInference(resized_img);

    detect_result_group_t detect_result_group;
    post_process(
        static_cast<int8_t*>(output.buffers[0]),
        static_cast<int8_t*>(output.buffers[1]),
        static_cast<int8_t*>(output.buffers[2]),
        height, width,
        BOX_THRESH, NMS_THRESH,
        scale, scale,
        output.zps, output.scales,
        &detect_result_group
    );

    detections.clear();
    int img_width = frame.cols;
    int img_height = frame.rows;

    for (int i = 0; i < detect_result_group.count; i++) {
        detect_result_t* det_result = &detect_result_group.results[i];

        if (!targetClasses.empty() && 
            find(targetClasses.begin(), targetClasses.end(), det_result->name) == targetClasses.end()) {
            continue;
        }

        int x1 = clamp(static_cast<int>((det_result->box.left - dx) / scale), 0, img_width - 1);
        int y1 = clamp(static_cast<int>((det_result->box.top - dy) / scale), 0, img_height - 1);
        int x2 = clamp(static_cast<int>((det_result->box.right - dx) / scale), 0, img_width - 1);
        int y2 = clamp(static_cast<int>((det_result->box.bottom - dy) / scale), 0, img_height - 1);

        Detection det;
        det.classId = det_result->name;
        det.confidence = det_result->prop;
        det.box = Rect(x1, y1, x2 - x1, y2 - y1);
        detections.push_back(det);
    }

    drawDetections(frame, detections);
    releaseOutputs(output);
}

const vector<Detection>& GenericDetector::getDetections() const {
    return detections;
}

int GenericDetector::clamp(int val, int min_val, int max_val) {
    return max(min_val, min(val, max_val));
}

// Utility function implementations
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

void printUsage(const string& programName) {
    cerr << "Usage: " << programName << " [--video <video_file>] [--image <image_file>] [--model <model_path>]" << endl;
}
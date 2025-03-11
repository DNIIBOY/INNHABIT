#include "detector.h"
#include <opencv2/dnn.hpp>
#include <opencv2/cudawarping.hpp>  // For cuda::resize
#include <opencv2/cudaimgproc.hpp>  // For cuda::cvtColor
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace cv;
using namespace cv::dnn;
using namespace std;

#ifndef BOX_THRESH
#define BOX_THRESH 0.25
#endif
#ifndef NMS_THRESH
#define NMS_THRESH 0.45
#endif

class JetsonDetector : public GenericDetector {
private:
    Net net;
    vector<string> classNames;

    void preprocessFrame(const Mat& frame, Mat& blob);
    void runInferenceGPU(const Mat& blob, vector<Mat>& outs);
    void postprocessDetections(const vector<Mat>& outs, Mat& frame, float scale, int dx, int dy);

public:
    JetsonDetector(const string& modelPath, const vector<string>& targetClasses_)
        : GenericDetector(modelPath, targetClasses_) {
        initialize(modelPath);
    }

    void detect(Mat& frame) override {
        if (!initialized) {
            cerr << "Error: Detector not properly initialized" << endl;
            return;
        }
        Mat blob;
        preprocessFrame(frame, blob);

        vector<Mat> outs;
        runInferenceGPU(blob, outs);

        int img_width = frame.cols;
        int img_height = frame.rows;
        float scale = min(static_cast<float>(width) / img_width, static_cast<float>(height) / img_height);
        int new_width = static_cast<int>(img_width * scale);
        int new_height = static_cast<int>(img_height * scale);
        int dx = (width - new_width) / 2;
        int dy = (height - new_height) / 2;

        postprocessDetections(outs, frame, scale, dx, dy);
    }

protected:
    void initialize(const string& modelPath) override {
        string namesFile = modelPath + "/coco.names";
        ifstream ifs(namesFile);
        if (!ifs.is_open()) {
            cerr << "Error: Could not open " << namesFile << endl;
            throw runtime_error("Failed to load class names");
        }
        string line;
        while (getline(ifs, line)) {
            if (!line.empty()) classNames.push_back(line);
        }
        ifs.close();

        string cfg = modelPath + "/yolov7-tiny.cfg";
        string weights = modelPath + "/yolov7-tiny.weights";
        net = readNet(cfg, weights);
#ifdef USE_CUDA
        net.setPreferableBackend(DNN_BACKEND_CUDA);
        net.setPreferableTarget(DNN_TARGET_CUDA_FP16);
#else
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
#endif

        if (net.empty()) {
            throw runtime_error("Failed to load YOLOv7-tiny model");
        }

        width = 320;
        height = 320;
        channel = 3;
        initialized = true;
    }

    DetectionOutput runInference(const Mat& input) override {
        DetectionOutput output;
        output.num_outputs = 0;
        return output;
    }

    ~JetsonDetector() override {
    }
};

void JetsonDetector::preprocessFrame(const Mat& frame, Mat& blob) {
    // Upload frame to GPU
    cuda::GpuMat d_frame(frame);
    cuda::GpuMat d_img, d_resized_img, d_resized_part;

    // Convert to RGB on GPU
    cuda::cvtColor(d_frame, d_img, COLOR_BGR2RGB);

    // Resize with letterboxing on GPU
    int img_width = d_img.cols;
    int img_height = d_img.rows;
    float scale = min(static_cast<float>(width) / img_width, static_cast<float>(height) / img_height);
    int new_width = static_cast<int>(img_width * scale);
    int new_height = static_cast<int>(img_height * scale);
    int dx = (width - new_width) / 2;
    int dy = (height - new_height) / 2;

    d_resized_img = cuda::GpuMat(height, width, CV_8UC3, Scalar(114, 114, 114));
    cuda::resize(d_img, d_resized_part, Size(new_width, new_height));
    d_resized_part.copyTo(d_resized_img(Rect(dx, dy, new_width, new_height)));

    // Download to CPU and create 4D blob
    Mat resized_img;
    d_resized_img.download(resized_img);
    blob = blobFromImage(resized_img, 1.0 / 255.0, Size(width, height), Scalar(0, 0, 0), true, false);
}

void JetsonDetector::runInferenceGPU(const Mat& blob, vector<Mat>& outs) {
    net.setInput(blob);
    vector<String> outNames = net.getUnconnectedOutLayersNames();
    net.forward(outs, outNames);
}

void JetsonDetector::postprocessDetections(const vector<Mat>& outs, Mat& frame, float scale, int dx, int dy) {
    int img_width = frame.cols;
    int img_height = frame.rows;

    vector<Rect> boxes;
    vector<float> confidences;
    vector<int> classIds;

    for (size_t i = 0; i < outs.size(); ++i) {
        float* data = (float*)outs[i].data;
        int rows = outs[i].rows;
        int cols = outs[i].cols;

        for (int j = 0; j < rows; ++j) {
            float confidence = data[j * cols + 4];
            if (confidence > BOX_THRESH) {
                Mat scores = outs[i].row(j).colRange(5, cols);
                Point classIdPoint;
                double maxScore;
                minMaxLoc(scores, 0, &maxScore, 0, &classIdPoint);

                float totalConfidence = confidence * maxScore;
                if (totalConfidence > BOX_THRESH) {
                    float centerX = data[j * cols + 0] * width;
                    float centerY = data[j * cols + 1] * height;
                    float w = data[j * cols + 2] * width;
                    float h = data[j * cols + 3] * height;

                    int left = (centerX - w / 2 - dx) / scale;
                    int top = (centerY - h / 2 - dy) / scale;
                    int boxWidth = w / scale;
                    int boxHeight = h / scale;

                    left = max(0, min(left, img_width - 1));
                    top = max(0, min(top, img_height - 1));
                    boxWidth = min(boxWidth, img_width - left);
                    boxHeight = min(boxHeight, img_height - top);

                    boxes.push_back(Rect(left, top, boxWidth, boxHeight));
                    confidences.push_back(totalConfidence);
                    classIds.push_back(classIdPoint.x);
                }
            }
        }
    }

    vector<int> indices;
    NMSBoxes(boxes, confidences, BOX_THRESH, NMS_THRESH, indices);

    detections.clear();
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        int classId = classIds[idx];
        string className = (classId >= 0 && classId < classNames.size()) ? classNames[classId] : "unknown";

        if (!targetClasses.empty()) {
            bool isTarget = false;
            for (const auto& target : targetClasses) {
                if (className == target) {
                    isTarget = true;
                    break;
                }
            }
            if (!isTarget) continue;
        }

        Rect box = boxes[idx];
        Detection det;
        det.classId = className;
        det.confidence = confidences[idx];
        det.box = box;
        detections.push_back(det);
    }
}

#ifdef USE_CUDA
Detector* createDetector(const string& modelPath, const vector<string>& targetClasses) {
    try {
        return new JetsonDetector(modelPath, targetClasses);
    } catch (const exception& e) {
        cerr << "Error creating Jetson detector: " << e.what() << endl;
        return nullptr;
    }
}
#else
Detector* createDetector(const string&, const vector<string>&) { return nullptr; }
#endif
#include "detector.h"
#include "common.h"
#include <opencv2/dnn.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>

class JetsonDetector : public GenericDetector {
private:
    cv::dnn::Net net;

public:
    JetsonDetector(const string& modelPath, const vector<string>& targetClasses_)
        : GenericDetector(modelPath, targetClasses_) {
        initialize(modelPath);
    }

    void detect(Mat& frame) override {
        if (!initialized) {
            ERROR("Detector not properly initialized");
            return;
        }

        float scale;
        int dx, dy;
        Mat resized_img = preprocessImage(frame, width, height, scale, dx, dy);
        Mat blob = cv::dnn::blobFromImage(resized_img, 1.0 / 255.0, Size(width, height), Scalar(0, 0, 0), true, false);

        net.setInput(blob);
        vector<Mat> outs;
        net.forward(outs, net.getUnconnectedOutLayersNames());

        detections.clear();
        processDetections(outs, frame, scale, dx, dy);
        drawDetections(frame, detections);
    }

protected:
    void initialize(const string& modelPath) override {
        string cfg = modelPath + "/yolov7-tiny.cfg";
        string weights = modelPath + "/yolov7-tiny.weights";
        net = cv::dnn::readNet(cfg, weights);
        
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);

        if (net.empty()) throw runtime_error("Failed to load YOLOv7-tiny model");
        initialized = true;
    }

    void processDetections(const vector<Mat>& outs, Mat& frame, float scale, int dx, int dy) {
        vector<Rect> boxes;
        vector<float> confidences;
        vector<int> classIds;

        for (const auto& out : outs) {
            float* data = (float*)out.data;
            for (int j = 0; j < out.rows; ++j) {
                float confidence = data[j * out.cols + 4];
                if (confidence > BOX_THRESH) {
                    Mat scores = out.row(j).colRange(5, out.cols);
                    Point classIdPoint;
                    double maxScore;
                    minMaxLoc(scores, 0, &maxScore, 0, &classIdPoint);

                    float totalConfidence = confidence * maxScore;
                    if (totalConfidence > BOX_THRESH) {
                        float centerX = data[j * out.cols + 0] * width;
                        float centerY = data[j * out.cols + 1] * height;
                        float w = data[j * out.cols + 2] * width;
                        float h = data[j * out.cols + 3] * height;

                        int left = (centerX - w / 2 - dx) / scale;
                        int top = (centerY - h / 2 - dy) / scale;
                        int boxWidth = w / scale;
                        int boxHeight = h / scale;

                        boxes.push_back(Rect(left, top, boxWidth, boxHeight));
                        confidences.push_back(totalConfidence);
                        classIds.push_back(classIdPoint.x);
                    }
                }
            }
        }

        vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, BOX_THRESH, NMS_THRESH, indices);

        int img_width = frame.cols;
        int img_height = frame.rows;
        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            // Use COCO_LABELS from common.h instead of generic "class_" prefix
            string className = classIds[idx] < 80 ? COCO_LABELS[classIds[idx]] : "unknown";
            if (!targetClasses.empty() && find(targetClasses.begin(), targetClasses.end(), className) == targetClasses.end()) continue;

            Detection det;
            det.classId = className;
            det.confidence = confidences[idx];
            det.box = boxes[idx];
            det.box.x = clamp(det.box.x, 0, img_width - 1);
            det.box.y = clamp(det.box.y, 0, img_height - 1);
            det.box.width = clamp(det.box.width, 0, img_width - det.box.x);
            det.box.height = clamp(det.box.height, 0, img_height - det.box.y);
            detections.push_back(det);
        }
    }

    DetectionOutput runInference(const Mat&) override { return DetectionOutput(); } // Not used

    void releaseOutputs(const DetectionOutput&) override {
        // No-op: JetsonDetector doesn't use DetectionOutput buffers
    }
};

Detector* createDetector(const string& modelPath, const vector<string>& targetClasses) {
    try {
        return new JetsonDetector(modelPath, targetClasses);
    } catch (const exception& e) {
        ERROR("Error creating Jetson detector: " << e.what());
        return nullptr;
    }
}

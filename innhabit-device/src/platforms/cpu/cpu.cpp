#include "detector.h"
#include "common.h"
#include <opencv2/dnn.hpp>

class CPUDetector : public GenericDetector {
private:
    cv::dnn::Net net;
    vector<string> classes;

public:
    CPUDetector(const string& modelPath, const vector<string>& targetClasses_)
        : GenericDetector(modelPath, targetClasses_) {
        initialize(modelPath);
    }

protected:
    void initialize(const string& modelPath) override {
        string cfg = modelPath + "/yolov7-tiny.cfg";
        string weights = modelPath + "/yolov7-tiny.weights";
        net = cv::dnn::readNet(cfg, weights);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        cv::setNumThreads(cv::getNumThreads());  // Maximize thread usage
        if (net.empty()) throw runtime_error("Failed to load YOLOv7 model for CPU");
        initialized = true;
    }

    DetectionOutput runInference(const Mat& input) override {
        Mat blob = cv::dnn::blobFromImage(input, 1/255.0, Size(width, height), Scalar(0,0,0), true, false);
        net.setInput(blob);
        
        DetectionOutput output;
        net.forward(output.outputs, net.getUnconnectedOutLayersNames());
        output.num_outputs = output.outputs.size();
        
        return output;
    }

    void processDetections(const DetectionOutput& output, Mat& frame, float scale, int dx, int dy) override {
        vector<Rect> boxes;
        vector<float> confidences;
        vector<int> classIds;

        for (const auto& out : output.outputs) {
            float* data = (float*)out.data;
            for (int j = 0; j < out.rows; ++j) {
                Mat scores = out.row(j).colRange(5, out.cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

                if (confidence > BOX_THRESH) {
                    int centerX = (int)(data[j * out.cols + 0] * frame.cols);
                    int centerY = (int)(data[j * out.cols + 1] * frame.rows);
                    int width = (int)(data[j * out.cols + 2] * frame.cols);
                    int height = (int)(data[j * out.cols + 3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }

        vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, BOX_THRESH, NMS_THRESH, indices);

        detections.clear();
        int img_width = frame.cols;
        int img_height = frame.rows;
        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
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

    void releaseOutputs(const DetectionOutput& output) override {
        // No-op: CPU detector doesn't need to release memory
    }
};

Detector* createDetector(const string& modelPath, const vector<string>& targetClasses) {
    try {
        return new CPUDetector(modelPath, targetClasses);
    } catch (const exception& e) {
        ERROR("Error creating CPU detector: " << e.what());
        return nullptr;
    }
}
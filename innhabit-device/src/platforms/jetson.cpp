#include "detector.h"
#include "common.h"
#include <opencv2/dnn.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

class JetsonDetector : public GenericDetector {
private:
    cv::dnn::Net net;
    cv::cuda::GpuMat gpu_frame;     // Reusable GPU buffers
    cv::cuda::GpuMat gpu_resized;
    
public:
    JetsonDetector(const string& modelPath, const vector<string>& targetClasses_)
        : GenericDetector(modelPath, targetClasses_) {
        initialize(modelPath);
    }

protected:
    void initialize(const string& modelPath) override {
        // Use environment variable to control thread count
        setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1", 1);
        
        string cfg = modelPath + "/yolov7-tiny.cfg";
        string weights = modelPath + "/yolov7-tiny.weights";
        net = cv::dnn::readNet(cfg, weights);
        
        // Configure for CUDA backend - use FP32 for better stability
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA); // Use FP32 instead of FP16 for reliability
        
        if (net.empty()) throw runtime_error("Failed to load YOLOv7-tiny model");
        initialized = true;
    }

    DetectionOutput runInference(const Mat& input) override {
        // Create blob directly from CPU Mat - this is more reliable
        Mat blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, Size(width, height), Scalar(0, 0, 0), true, false);
        
        // Set input to network
        net.setInput(blob);
        
        // Run inference
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
                float confidence = data[j * out.cols + 4];
                if (confidence > BOX_THRESH) {
                    Mat scores = out.row(j).colRange(5, out.cols);
                    Point classIdPoint;
                    double maxScore;
                    minMaxLoc(scores, 0, &maxScore, 0, &classIdPoint);

                    float totalConfidence = confidence * maxScore;
                    if (totalConfidence > BOX_THRESH) {
                        // Use width/height with input image dimensions
                        float centerX = data[j * out.cols + 0] * width;
                        float centerY = data[j * out.cols + 1] * height;
                        float w = data[j * out.cols + 2] * width;
                        float h = data[j * out.cols + 3] * height;

                        // Calculate box coordinates
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

        // Use NMS to reduce overlapping boxes
        vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, BOX_THRESH, NMS_THRESH, indices);

        // Create detection objects from filtered boxes
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
            
            // Clamp box coordinates to image boundaries
            det.box.x = clamp(det.box.x, 0, img_width - 1);
            det.box.y = clamp(det.box.y, 0, img_height - 1);
            det.box.width = clamp(det.box.width, 0, img_width - det.box.x);
            det.box.height = clamp(det.box.height, 0, img_height - det.box.y);
            
            detections.push_back(det);
        }
    }

    void releaseOutputs(const DetectionOutput& output) override {
        // CUDA resources are managed by OpenCV
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
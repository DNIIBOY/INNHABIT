#include "detector.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

#define ERROR(x) std::cerr << x << std::endl

// Implementation of CPUDetector
class CPUDetector : public GenericDetector {
private:
    // Model parameters
    int model_input_w_; // Model input width
    int model_input_h_; // Model input height
    int num_detections_; // Number of detections (e.g., 8400 for YOLO)
    int detection_attribute_size_; // Detection attribute size (e.g., 4 + 1 + num_classes)
    int num_classes_; // Number of classes
    cv::dnn::Net net_; // OpenCV DNN model
    cv::Mat input_image; // Store original input image for scaling
    std::vector<cv::Mat> outputs; // Store inference outputs
    DetectionOutput detection_output_; // Store platform-specific outputs

    void preprocess(cv::Mat& frame) override {
        input_image = frame.clone(); // Store original image for postprocessing
        // Resize and normalize the frame for YOLOv11
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(model_input_w_, model_input_h_), 0, 0, cv::INTER_LINEAR);
        // Convert to float and normalize to [0, 1]
        resized.convertTo(resized, CV_32F, 1.0 / 255.0);
        // Ensure the frame is in the correct format (BGR, 3 channels)
        if (resized.channels() == 4) {
            cv::cvtColor(resized, resized, cv::COLOR_BGRA2BGR);
        }
        frame = resized; // Update frame for inference
    }

    void inference(cv::Mat& frame) override {
        // Create blob from preprocessed frame
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(model_input_w_, model_input_h_),
                                             cv::Scalar(0, 0, 0), true, false);
        net_.setInput(blob);
        outputs.clear();
        net_.forward(outputs, net_.getUnconnectedOutLayersNames());
        detection_output_.outputs = outputs;
    }

    void postprocess(cv::Mat& frame) override {
        // Scaling factors (adopted from TensorRTDetector)
        const float ratio_h = static_cast<float>(model_input_h_) / input_image.rows;
        const float ratio_w = static_cast<float>(model_input_w_) / input_image.cols;
        const float scale = std::min(ratio_h, ratio_w);
        const float offset_x = (ratio_h > ratio_w) ? 0.0f : (model_input_w_ - ratio_h * input_image.cols) * 0.5f;
        const float offset_y = (ratio_h > ratio_w) ? (model_input_h_ - ratio_w * input_image.rows) * 0.5f : 0.0f;
    
        // Debug output shape
        std::cout << "Output shape: [";
        for (int i = 0; i < outputs[0].size.dims(); ++i) {
            std::cout << outputs[0].size[i] << (i < outputs[0].size.dims() - 1 ? ", " : "]");
        }
        std::cout << std::endl;
    
        // Assuming outputs[0] is [1, num_detections, detection_attribute_size]
        float* data = outputs[0].ptr<float>();
        num_detections_ = outputs[0].size[1];
        detection_attribute_size_ = outputs[0].size[2];
        num_classes_ = detection_attribute_size_ - 5; // 4 for box, 1 for confidence
    
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        boxes.reserve(100); // Reserve capacity
        confidences.reserve(100);
        class_ids.reserve(100);
    
        // Parse detections
        for (int i = 0; i < num_detections_; ++i) {
            float* detection = data + i * detection_attribute_size_;
            float confidence = detection[4]; // Objectness score
            if (confidence >= 0.4) {
                // Simplified for single-class (adjust if multi-class)
                int max_class_id = 0; // Assume single class
                float class_score = confidence; // No class score multiplication
                if (class_score >= 0.25) {
                    // Extract bounding box
                    float center_x = detection[0];
                    float center_y = detection[1];
                    float width = detection[2];
                    float height = detection[3];
    
                    // Scale and clamp
                    int x = static_cast<int>((center_x - 0.5f * width - offset_x) / scale);
                    int y = static_cast<int>((center_y - 0.5f * height - offset_y) / scale);
                    int w = static_cast<int>(width / scale);
                    int h = static_cast<int>(height / scale);
    
                    x = clamp(x, 0, input_image.cols - 1);
                    y = clamp(y, 0, input_image.rows - 1);
                    w = clamp(w, 0, input_image.cols - x);
                    h = clamp(h, 0, input_image.rows - y);
    
                    try {
                        boxes.emplace_back(x, y, w, h);
                        confidences.push_back(class_score);
                        class_ids.push_back(max_class_id);
                    } catch (const std::bad_alloc& e) {
                        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
                        return;
                    }
                }
            }
        }
    
        // Apply NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indices);
    
        // Store final detections
        detections_.clear();
        detections_.reserve(indices.size());
        for (int idx : indices) {
            Detection det;
            det.bounding_box = boxes[idx];
            det.confidence = confidences[idx];
            det.class_id = (idx < target_classes_.size()) ? target_classes_[class_ids[idx]] : std::to_string(class_ids[idx]);
            detections_.push_back(det);
        }
    }

    void releaseOutputs() override {
        detection_output_.clear();
        outputs.clear();
    }

public:
    CPUDetector(const string& model_path, const vector<string>& target_classes)
        : GenericDetector(model_path, target_classes), num_classes_(static_cast<int>(target_classes.size())) {
        initialize(model_path);
    }

    ~CPUDetector() override {
        releaseOutputs();
    }

    void initialize(const string& model_path) override {
        try {
            net_ = cv::dnn::readNet(model_path);
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

            // Set input dimensions for YOLOv11 (default 640x640 for YOLOv11)
            model_input_w_ = 512; // YOLOv11 typically uses 640x640
            model_input_h_ = 512;

            cv::setNumThreads(cv::getNumThreads());

            if (net_.empty()) {
                ERROR("Failed to load model: " << model_path);
                return;
            }
            initialized_ = true;
        } catch (const std::exception& e) {
            ERROR("Failed to initialize model: " << e.what());
            initialized_ = false;
        }
    }
};

GenericDetector* createDetector(const string& model_path, const vector<string>& target_classes) {
    try {
        return new CPUDetector(model_path, target_classes);
    } catch (const std::exception& e) {
        ERROR("Error creating CPU detector: " << e.what());
        return nullptr;
    }
}
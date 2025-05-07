#include "detector.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

#define ERROR(x) std::cerr << x << std::endl

// Implementation of GenericDetector methods
GenericDetector::GenericDetector(const string& model_path, const vector<string>& target_classes)
    : target_classes_(target_classes) {
}

int GenericDetector::clamp(int value, int min_value, int max_value) {
    return std::max(min_value, std::min(value, max_value));
}

void GenericDetector::detect(cv::Mat& frame) {
    if (!initialized_) {
        ERROR("Detector not initialized");
        return;
    }
    detections_.clear();
    preprocess(frame);
    inference(frame);
    postprocess(frame);
}

// Implementation of CPUDetector
class CPUDetector : public GenericDetector {
private:
    // Model parameters
    int model_input_w_; // Model input width
    int model_input_h_; // Model input height
    int num_detections_; // Number of detections (e.g., 8400 for YOLO)
    int detection_attribute_size_; // Detection attribute size (e.g., 4 + 1 + num_classes)
    int num_classes_; // Number of classes (e.g., 1 or 80 for COCO)
    cv::dnn::Net net_; // OpenCV DNN model
    cv::Mat input_image; // Store original input image for scaling
    std::vector<cv::Mat> outputs; // Store inference outputs
    DetectionOutput detection_output_; // Store platform-specific outputs

    void preprocess(cv::Mat& frame) override {
        input_image = frame.clone(); // Store original image for postprocessing
        // Resize and normalize the frame
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(model_input_w_, model_input_h_), 0, 0, cv::INTER_LINEAR);
        // Convert to float and normalize to [0, 1]
        resized.convertTo(resized, CV_32F, 1.0 / 255.0);
        // Ensure the frame is in the correct format (BGR, 3 channels)
        if (resized.channels() == 4) {
            cv::cvtColor(resized, resized, cv::COLOR_BGRA2BGR);
        }
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
        // Scaling factors to map bounding boxes back to original image size
        float x_factor = input_image.cols / static_cast<float>(model_input_w_);
        float y_factor = input_image.rows / static_cast<float>(model_input_h_);

        // Assuming outputs[0] is [1, num_detections, detection_attribute_size]
        float* data = outputs[0].ptr<float>();
        num_detections_ = outputs[0].size[1];
        detection_attribute_size_ = outputs[0].size[2];
        num_classes_ = detection_attribute_size_ - 5; // 4 for box, 1 for confidence

        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        // Parse detections
        for (int i = 0; i < num_detections_; ++i) {
            float* detection = data + i * detection_attribute_size_;
            float confidence = detection[4]; // Objectness score
            if (confidence >= 0.4) { // Confidence threshold
                // Find class with maximum score
                float max_class_score = 0;
                int max_class_id = 0;
                for (int j = 5; j < detection_attribute_size_; ++j) {
                    if (detection[j] > max_class_score) {
                        max_class_score = detection[j];
                        max_class_id = j - 5;
                    }
                }
                float class_score = confidence * max_class_score;
                if (class_score >= 0.25) { // Combined confidence threshold
                    // Extract bounding box (center_x, center_y, width, height)
                    float center_x = detection[0];
                    float center_y = detection[1];
                    float width = detection[2];
                    float height = detection[3];

                    // Scale to original image size
                    int x = static_cast<int>((center_x - width / 2) * x_factor);
                    int y = static_cast<int>((center_y - height / 2) * y_factor);
                    int w = static_cast<int>(width * x_factor);
                    int h = static_cast<int>(height * y_factor);

                    // Clamp coordinates to image boundaries
                    x = clamp(x, 0, input_image.cols - 1);
                    y = clamp(y, 0, input_image.rows - 1);
                    w = clamp(w, 0, input_image.cols - x);
                    h = clamp(h, 0, input_image.rows - y);

                    boxes.emplace_back(x, y, w, h);
                    confidences.push_back(class_score);
                    class_ids.push_back(max_class_id);
                }
            }
        }

        // Apply Non-Maximum Suppression (NMS)
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indices);

        // Store final detections
        detections_.clear();
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

            // Set default input dimensions (adjust based on your model, e.g., YOLOv8 uses 640x640)
            model_input_w_ = 512;
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
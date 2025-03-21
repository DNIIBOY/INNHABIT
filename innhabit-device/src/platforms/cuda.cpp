#include "detector.h"
#include "common.h"
#include <opencv2/dnn.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

class JetsonDetector : public GenericDetector {
private:
    cv::dnn::Net net;
    cv::cuda::GpuMat gpu_frame;
    cv::cuda::GpuMat gpu_rgb;
    cv::cuda::GpuMat gpu_resized_part;
    cv::cuda::GpuMat gpu_output;
    cv::cuda::Stream stream;
    Mat resized_img;  // Added to store preprocessing result

    const Scalar PADDING_COLOR = Scalar(114, 114, 114);
    float scale;  // Added to store preprocessing parameters
    int dx, dy;

    DetectionOutput output;

    static constexpr int targetWidth = 640;
    static constexpr int targetHeight = 640;

public:
    JetsonDetector(const string& modelPath, const vector<string>& targetClasses_)
        : GenericDetector(modelPath, targetClasses_) {
        initialize(modelPath);
        gpu_frame.create(720, 1280, CV_8UC3);
        gpu_rgb.create(720, 1280, CV_8UC3);
    }

    ~JetsonDetector() override {
        gpu_frame.release();
        gpu_rgb.release();
        gpu_resized_part.release();
        gpu_output.release();
    }

    void printDetections() const {
        size_t displayCount = std::min<size_t>(detections_.size(), 3);
        for (size_t i = 0; i < displayCount; ++i) {
            const Detection& det = detections_[i];
            std::cout << "  Detection " << i + 1 << ": Class=" << det.class_id 
                      << ", Conf=" << det.confidence << ", Box=[x=" << det.bounding_box.x 
                      << ",y=" << det.bounding_box.y << ",w=" << det.bounding_box.width 
                      << ",h=" << det.bounding_box.height << "]" << std::endl;
        }
        if (detections_.size() > displayCount) {
            std::cout << "... and " << (detections_.size() - displayCount) << " more" << std::endl;
        }
    }

protected:
    void initialize(const string& modelPath) override {
        setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1", 1);
        
        string cfg = modelPath + "/yolov7-tiny.cfg";
        string weights = modelPath + "/yolov7-tiny.weights";

        try {
            net = cv::dnn::readNet(cfg, weights);
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            if (net.empty()) throw runtime_error("Failed to load YOLOv7-tiny model");
            initialized_ = true;
        } catch (const exception& e) {
            ERROR("Error initializing detector: " << e.what());
            initialized_ = false;
        }
    }

    void preprocess(cv::Mat& frame) override {
        gpu_frame.upload(frame, stream);
    
        // Convert to RGB format on GPU
        cv::cuda::cvtColor(gpu_frame, gpu_rgb, COLOR_BGR2RGB, 0, stream);
    
        // Calculate scaling and padding parameters
        const int img_width = frame.cols;
        const int img_height = frame.rows;
        scale = min(static_cast<float>(targetWidth) / img_width, static_cast<float>(targetHeight) / img_height);
        const int new_width = static_cast<int>(img_width * scale);
        const int new_height = static_cast<int>(img_height * scale);
        dx = (targetWidth - new_width) / 2;
        dy = (targetHeight - new_height) / 2;
    
        // Resize the image on the GPU
        cv::cuda::resize(gpu_rgb, gpu_resized_part, Size(new_width, new_height), 0, 0, INTER_LINEAR, stream);
    
        // Prepare the output frame with padding
        gpu_output.create(targetHeight, targetWidth, CV_8UC3);
        gpu_output.setTo(PADDING_COLOR, stream);
        gpu_resized_part.copyTo(gpu_output(cv::Rect(dx, dy, new_width, new_height)), stream);
    
        // Download the processed image from GPU to CPU
        gpu_output.download(resized_img, stream);
        stream.waitForCompletion();
    }

    void inference(cv::Mat& input) override {
        // Create a blob from the input image
        Mat blob = cv::dnn::blobFromImage(input, 1.0/255.0, Size(targetWidth, targetHeight), 
                                          Scalar(0, 0, 0), true, false);
    
        // Set the input for the network
        net.setInput(blob);
    
        // Perform inference and retrieve the output
        static const auto outLayerNames = net.getUnconnectedOutLayersNames();
        net.forward(output.outputs, outLayerNames);
    }
    

    void postprocess(cv::Mat& frame) override {
        const int img_width = frame.cols;
        const int img_height = frame.rows;
    
        std::vector<Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> classIds;
    
        boxes.reserve(100);
        confidences.reserve(100);
        classIds.reserve(100);
    
        // Process each output from the inference
        for (const auto& out : output.outputs) {
            const float* data = (float*)out.data;
            const int dimensions = out.cols;
    
            for (int j = 0; j < out.rows; ++j) {
                const float confidence = data[j * dimensions + 4];
                if (confidence > BOX_THRESH) {
                    // Get the class with the highest score
                    const Mat scores = out.row(j).colRange(5, dimensions);
                    Point classIdPoint;
                    double maxScore;
                    minMaxLoc(scores, 0, &maxScore, 0, &classIdPoint);
    
                    const float totalConfidence = confidence * maxScore;
                    if (totalConfidence > BOX_THRESH) {
                        const float centerX = data[j * dimensions + 0] * targetWidth;
                        const float centerY = data[j * dimensions + 1] * targetHeight;
                        const float w = data[j * dimensions + 2] * targetWidth;
                        const float h = data[j * dimensions + 3] * targetWidth;
    
                        const float x_resized = centerX - w / 2;
                        const float y_resized = centerY - h / 2;
    
                        // Undo padding and scaling
                        const int left = static_cast<int>((x_resized - dx) / scale);
                        int top = static_cast<int>((y_resized - dy) / scale);
                        const int boxWidth = static_cast<int>(w / scale);
                        const int boxHeight = static_cast<int>(h / scale);
    
                        // Ensure 'top' is not negative
                        top = std::max(top, 0);
    
                        // Ensure 'left' is not negative
                        const int clampedLeft = std::max(left, 0);
    
                        // Make sure the bounding box doesn't go out of image bounds
                        const int clampedBoxWidth = std::min(boxWidth, img_width - clampedLeft);
                        const int clampedBoxHeight = std::min(boxHeight, img_height - top);
    
                        boxes.push_back(Rect(clampedLeft, top, clampedBoxWidth, clampedBoxHeight));
                        confidences.push_back(totalConfidence);
                        classIds.push_back(classIdPoint.x);
                    }
                }
            }
        }
    
        // Apply non-maximum suppression (NMS)
        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, confidences, BOX_THRESH, NMS_THRESH, nms_result);
    
        // Build detections list
        for (size_t i = 0; i < nms_result.size(); ++i) {
            const int idx = nms_result[i];
            const int classId = classIds[idx];
            const string className = classId < 80 ? COCO_LABELS[classId] : "unknown";
    
            // Skip classes not in the target list
            if (!target_classes_.empty() && 
                find(target_classes_.begin(), target_classes_.end(), className) == target_classes_.end()) 
                continue;
    
            Detection det;
            det.class_id = className;
            det.confidence = confidences[idx];
            det.bounding_box = boxes[idx];
    
            // Apply clamping again to ensure the box is within image bounds
            det.bounding_box.x = clamp(det.bounding_box.x, 0, img_width - 1);
            det.bounding_box.y = clamp(det.bounding_box.y, 0, img_height - 1);
            det.bounding_box.width = clamp(det.bounding_box.width, 0, img_width - det.bounding_box.x);
            det.bounding_box.height = clamp(det.bounding_box.height, 0, img_height - det.bounding_box.y);
    
            detections_.push_back(det);
        }
    
        boxes.clear();
        confidences.clear();
        classIds.clear();
        nms_result.clear();
    }
    
    
    

    void releaseOutputs() override {
        detections_.clear();
    }
};

GenericDetector* createDetector(const string& modelPath, const vector<string>& targetClasses) {
    try {
        return new JetsonDetector(modelPath, targetClasses);
    } catch (const exception& e) {
        ERROR("Error creating Jetson detector: " << e.what());
        return nullptr;
    }
}
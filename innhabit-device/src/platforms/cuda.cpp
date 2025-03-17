#include "detector.h"
#include "common.h"
#include <opencv2/dnn.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

class JetsonDetector : public GenericDetector {
private:
    cv::dnn::Net net;
    cv::cuda::GpuMat gpu_frame;     // Reusable GPU buffers
    cv::cuda::GpuMat gpu_rgb;
    cv::cuda::GpuMat gpu_resized_part;
    cv::cuda::GpuMat gpu_output;
    cv::cuda::Stream stream;        // CUDA stream for asynchronous operations

    // Constants moved to class scope to avoid reallocation
    const Scalar PADDING_COLOR = Scalar(114, 114, 114);
    vector<Rect> boxes;
    vector<float> confidences;
    vector<int> classIds;
    vector<int> indices;
    
    // GPU-optimized preprocessing with stream support
    Mat preprocessImage(const Mat& frame, int targetWidth, int targetHeight, float& scale, int& dx, int& dy) {
        // Upload frame to GPU with stream
        gpu_frame.upload(frame, stream);

        // Convert color space on GPU (BGR to RGB) with stream
        cv::cuda::cvtColor(gpu_frame, gpu_rgb, COLOR_BGR2RGB, 0, stream);

        // Calculate scaling and padding parameters
        const int img_width = frame.cols;
        const int img_height = frame.rows;
        scale = min(static_cast<float>(targetWidth) / img_width, static_cast<float>(targetHeight) / img_height);
        const int new_width = static_cast<int>(img_width * scale);
        const int new_height = static_cast<int>(img_height * scale);
        dx = (targetWidth - new_width) / 2;
        dy = (targetHeight - new_height) / 2;

        // Resize on GPU with stream
        cv::cuda::resize(gpu_rgb, gpu_resized_part, Size(new_width, new_height), 0, 0, INTER_LINEAR, stream);

        // Create padded output on GPU with padding color
        gpu_output.create(targetHeight, targetWidth, CV_8UC3);
        gpu_output.setTo(PADDING_COLOR, stream);

        // Copy resized part to padded output with stream
        gpu_resized_part.copyTo(gpu_output(cv::Rect(dx, dy, new_width, new_height)), stream);

        // Download result to CPU memory with stream
        Mat result;
        gpu_output.download(result, stream);
        
        // Wait for GPU operations to complete
        stream.waitForCompletion();
        
        return result;
    }

public:
    JetsonDetector(const string& modelPath, const vector<string>& targetClasses_)
        : GenericDetector(modelPath, targetClasses_) {
        initialize(modelPath);
        
        // Pre-allocate GPU memory for better performance
        gpu_frame.create(720, 1280, CV_8UC3);  // Typical camera resolution
        gpu_rgb.create(720, 1280, CV_8UC3);
    }

    ~JetsonDetector() {
        // Release CUDA resources explicitly
        gpu_frame.release();
        gpu_rgb.release();
        gpu_resized_part.release();
        gpu_output.release();
    }

    // Override detect to use GPU-optimized preprocessing
    void detect(Mat& frame) override {
        if (!initialized) {
            ERROR("Detector not properly initialized");
            return;
        }

        // Clear previous detection data
        boxes.clear();
        confidences.clear();
        classIds.clear();
        indices.clear();

        // Use GPU-optimized preprocessing
        float scale;
        int dx, dy;
        Mat resized_img = preprocessImage(frame, width, height, scale, dx, dy);

        // Run inference (CPU Mat passed)
        DetectionOutput output = runInference(resized_img);

        // Process detections
        processDetections(output, frame, scale, dx, dy);

        // Draw detections (common code from GenericDetector)
        //drawDetections(frame, detections);

        // Release outputs
        releaseOutputs(output);
    }

protected:
    void initialize(const string& modelPath) override {
        // Set environment variables for optimal CUDA performance
        setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1", 1);
        
        // Set CUDA device properties for better performance
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        if (prop.canMapHostMemory) {
            cudaSetDeviceFlags(cudaDeviceMapHost);
        }
        
        // Load model
        string cfg = modelPath + "/yolov7-tiny.cfg";
        string weights = modelPath + "/yolov7-tiny.weights";
        
        try {
            net = cv::dnn::readNet(cfg, weights);
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            
            // Enable FP16 inference for faster performance on Tensor cores
            //net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
            
            if (net.empty()) throw runtime_error("Failed to load YOLOv7-tiny model");
            initialized = true;
        } catch (const cv::Exception& e) {
            ERROR("OpenCV error: " << e.what());
            initialized = false;
        }
    }

    // Optimize blob creation and inference
    DetectionOutput runInference(const Mat& input) override {
        // Create blob directly on the GPU when possible
        Mat blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, Size(width, height), Scalar(0, 0, 0), true, false);
        
        // Set input to the network
        net.setInput(blob);
        
        DetectionOutput output;
        
        // Get output layer names once and reuse
        static const auto outLayerNames = net.getUnconnectedOutLayersNames();
        
        // Forward pass
        net.forward(output.outputs, outLayerNames);
        output.num_outputs = output.outputs.size();
        
        return output;
    }

    void processDetections(const DetectionOutput& output, Mat& frame, float scale, int dx, int dy) override {
        const int img_width = frame.cols;
        const int img_height = frame.rows;

        for (const auto& out : output.outputs) {
            const float* data = (float*)out.data;
            const int dimensions = out.cols;
            
            #pragma omp parallel for schedule(dynamic, 16)
            for (int j = 0; j < out.rows; ++j) {
                const float confidence = data[j * dimensions + 4];
                if (confidence > BOX_THRESH) {
                    // Create a Mat header without copying data
                    const Mat scores = out.row(j).colRange(5, dimensions);
                    Point classIdPoint;
                    double maxScore;
                    minMaxLoc(scores, 0, &maxScore, 0, &classIdPoint);

                    const float totalConfidence = confidence * maxScore;
                    if (totalConfidence > BOX_THRESH) {
                        const float centerX = data[j * dimensions + 0] * width;
                        const float centerY = data[j * dimensions + 1] * height;
                        const float w = data[j * dimensions + 2] * width;
                        const float h = data[j * dimensions + 3] * height;

                        const int left = (centerX - w / 2 - dx) / scale;
                        const int top = (centerY - h / 2 - dy) / scale;
                        const int boxWidth = w / scale;
                        const int boxHeight = h / scale;

                        #pragma omp critical
                        {
                            boxes.push_back(Rect(left, top, boxWidth, boxHeight));
                            confidences.push_back(totalConfidence);
                            classIds.push_back(classIdPoint.x);
                        }
                    }
                }
            }
        }

        // Non-maximum suppression
        cv::dnn::NMSBoxes(boxes, confidences, BOX_THRESH, NMS_THRESH, indices);

        // Build detections list
        detections.clear();
        for (size_t i = 0; i < indices.size(); ++i) {
            const int idx = indices[i];
            const int classId = classIds[idx];
            const string className = classId < 80 ? COCO_LABELS[classId] : "unknown";
            
            // Skip classes not in target list
            if (!targetClasses.empty() && 
                find(targetClasses.begin(), targetClasses.end(), className) == targetClasses.end()) 
                continue;

            // Create detection with bounds checking
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
        for (const auto& out : output.outputs) {
            cv::Mat tmp = out.clone();
            tmp.release();
        }
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
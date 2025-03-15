#include "detector.h"
#include "common.h"
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <memory>
#include <fstream>
#include <vector>
#include <chrono>

// Logger for TensorRT
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

// Helper function to load a serialized TensorRT engine
std::vector<unsigned char> loadEngine(const std::string& enginePath) {
    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        throw std::runtime_error("Failed to load TensorRT engine: " + enginePath);
    }
    
    file.seekg(0, std::ifstream::end);
    size_t size = file.tellg();
    file.seekg(0, std::ifstream::beg);
    
    std::vector<unsigned char> engineData(size);
    file.read(reinterpret_cast<char*>(engineData.data()), size);
    
    return engineData;
}

// CUDA memory management using RAII
class CudaBuffer {
    public:
        void* ptr = nullptr;
        size_t size = 0;
    
        CudaBuffer() = default;
    
        // Copy constructor to prevent shallow copy
        CudaBuffer(const CudaBuffer& other) {
            if (other.ptr) {
                allocate(other.size);
                cudaMemcpy(ptr, other.ptr, size, cudaMemcpyDeviceToDevice);
            }
        }
    
        // Assignment operator
        CudaBuffer& operator=(const CudaBuffer& other) {
            if (this != &other) {
                free();
                if (other.ptr) {
                    allocate(other.size);
                    cudaMemcpy(ptr, other.ptr, size, cudaMemcpyDeviceToDevice);
                }
            }
            return *this;
        }
    
        void allocate(size_t size_) {
            if (ptr) {
                free();
            }
            size = size_;
            cudaError_t err = cudaMalloc(&ptr, size);
            if (err != cudaSuccess || ptr == nullptr) {
                throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
            }
            LOG("Allocated buffer: ptr=" << ptr << ", size=" << size);
        }
    
        void free() {
            if (ptr) {
                cudaError_t err = cudaFree(ptr);
                if (err != cudaSuccess) {
                    ERROR("cudaFree failed: " << cudaGetErrorString(err));
                }
                ptr = nullptr;
                size = 0;
            }
        }
    
        ~CudaBuffer() {
            free();
        }
    };
// TensorRT Detector implementation
class TensorRTDetector : public GenericDetector {
private:
    // TensorRT objects
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    
    // CUDA buffers
    std::vector<CudaBuffer> inputBuffers;
    std::vector<CudaBuffer> outputBuffers;
    cudaStream_t stream;
    
    // Input dimensions
    int inputBindingIndex = -1;
    std::vector<int> outputBindingIndices;
    std::vector<size_t> outputSizes;
    
    // GPU resources
    cv::cuda::GpuMat gpu_frame;
    cv::cuda::GpuMat gpu_rgb;
    cv::cuda::GpuMat gpu_resized_part;
    cv::cuda::GpuMat gpu_output;
    
    // Detection buffers
    std::vector<Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;
    std::vector<int> indices;
    
    // Model specifics
    int num_classes = 80;
    int num_anchors = 0;
    bool is_yolov8 = true;  // Default to YOLOv8 format

    // Process YOLO detections from raw network output
    void processOutput(float* output, int outSize, int imgWidth, int imgHeight, float scale, int dx, int dy) {
        boxes.clear();
        confidences.clear();
        classIds.clear();
        
        if (is_yolov8) {
            // YOLOv8 format: [xywh, conf, class_scores]
            int dimensions = 4 + 1 + num_classes; // xywh + obj_conf + num_classes
            
            for (int i = 0; i < outSize / dimensions; ++i) {
                float* detection = &output[i * dimensions];
                
                float confidence = detection[4];
                if (confidence < BOX_THRESH) continue;
                
                // Find class with highest score
                float maxScore = 0;
                int maxClassId = 0;
                for (int c = 0; c < num_classes; ++c) {
                    float score = detection[5 + c];
                    if (score > maxScore) {
                        maxScore = score;
                        maxClassId = c;
                    }
                }
                
                float totalConfidence = confidence * maxScore;
                if (totalConfidence < BOX_THRESH) continue;
                
                // Denormalize box coordinates
                float x = detection[0];
                float y = detection[1];
                float w = detection[2];
                float h = detection[3];
                
                // Convert from center format to top-left format and scale
                int left = (x - w/2 - dx) / scale;
                int top = (y - h/2 - dy) / scale;
                int boxWidth = w / scale;
                int boxHeight = h / scale;
                
                boxes.push_back(Rect(left, top, boxWidth, boxHeight));
                confidences.push_back(totalConfidence);
                classIds.push_back(maxClassId);
            }
        } else {
            // YOLOv7-style format (anchor-based)
            int dimensions = 5 + num_classes;
            
            for (int i = 0; i < outSize / dimensions; ++i) {
                float* detection = &output[i * dimensions];
                
                float obj_conf = detection[4];
                if (obj_conf < BOX_THRESH) continue;
                
                // Find class with highest score
                float maxScore = 0;
                int maxClassId = 0;
                for (int c = 0; c < num_classes; ++c) {
                    float score = detection[5 + c] * obj_conf;
                    if (score > maxScore) {
                        maxScore = score;
                        maxClassId = c;
                    }
                }
                
                if (maxScore < BOX_THRESH) continue;
                
                // Denormalize box coordinates
                float x = detection[0] * width;
                float y = detection[1] * height;
                float w = detection[2] * width;
                float h = detection[3] * height;
                
                // Convert from center format to top-left format and scale
                int left = (x - w/2 - dx) / scale;
                int top = (y - h/2 - dy) / scale;
                int boxWidth = w / scale;
                int boxHeight = h / scale;
                
                boxes.push_back(Rect(left, top, boxWidth, boxHeight));
                confidences.push_back(maxScore);
                classIds.push_back(maxClassId);
            }
        }
    }
    
    // GPU-optimized preprocessing with stream support
    Mat preprocessImage(const Mat& frame, int targetWidth, int targetHeight, float& scale, int& dx, int& dy) {
        // Use default cv::cuda::Stream (OpenCV 4.10 compatible)
        cv::cuda::Stream cv_stream;
    
        // Upload frame to GPU
        if (gpu_frame.cols != frame.cols || gpu_frame.rows != frame.rows) {
            gpu_frame.create(frame.rows, frame.cols, CV_8UC3);
        }
        gpu_frame.upload(frame, cv_stream);
    
        // Convert to RGB
        if (gpu_rgb.cols != frame.cols || gpu_rgb.rows != frame.rows) {
            gpu_rgb.create(frame.rows, frame.cols, CV_8UC3);
        }
        cv::cuda::cvtColor(gpu_frame, gpu_rgb, cv::COLOR_BGR2RGB, 0, cv_stream);
    
        // Calculate scaling and padding
        const int img_width = frame.cols;
        const int img_height = frame.rows;
        scale = std::min(static_cast<float>(targetWidth) / img_width, static_cast<float>(targetHeight) / img_height);
        const int new_width = static_cast<int>(img_width * scale);
        const int new_height = static_cast<int>(img_height * scale);
        dx = (targetWidth - new_width) / 2;
        dy = (targetHeight - new_height) / 2;
    
        // Resize
        if (gpu_resized_part.cols != new_width || gpu_resized_part.rows != new_height) {
            gpu_resized_part.create(new_height, new_width, CV_8UC3);
        }
        cv::cuda::resize(gpu_rgb, gpu_resized_part, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR, cv_stream);
    
        // Pad
        if (gpu_output.cols != targetWidth || gpu_output.rows != targetHeight) {
            gpu_output.create(targetHeight, targetWidth, CV_8UC3);
        }
        gpu_output.setTo(cv::Scalar(114, 114, 114), cv_stream);
        gpu_resized_part.copyTo(gpu_output(cv::Rect(dx, dy, new_width, new_height)), cv_stream);
    
        // Download result
        Mat result;
        gpu_output.download(result, cv_stream);
    
        // Synchronize OpenCV operations
        cv_stream.waitForCompletion();
    
        // Synchronize with TensorRT stream
        cudaError_t err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaStreamSynchronize failed: " + std::string(cudaGetErrorString(err)));
        }
    
        return result;
    }

public:
    TensorRTDetector(const string& modelPath, const vector<string>& targetClasses_)
        : GenericDetector(modelPath, targetClasses_) {
        initialize(modelPath);
    }
    
    ~TensorRTDetector() {
        // Release TensorRT resources
        if (context) delete context;
        if (engine) delete engine;
        if (runtime) delete runtime;
        
        // Destroy CUDA stream
        cudaStreamDestroy(stream);
    }
    
    // Override detect to use GPU-optimized preprocessing
    void detect(Mat& frame) override {
        if (!initialized) {
            ERROR("TensorRT detector not properly initialized");
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

        // Run inference
        DetectionOutput output = runInference(resized_img);

        // Process detections
        processDetections(output, frame, scale, dx, dy);

        // Draw detections (common code from GenericDetector)
        drawDetections(frame, detections);

        // Release outputs
        releaseOutputs(output);
    }

protected:
    void initialize(const string& modelPath) override {
        cudaError_t err = cudaStreamCreate(&stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaStreamCreate failed: " + std::string(cudaGetErrorString(err)));
        }
        LOG("CUDA stream created: " << stream);
        gpu_frame.create(720, 1280, CV_8UC3);
        gpu_rgb.create(720, 1280, CV_8UC3);
        
        try {
            string enginePath = modelPath + "/yolov8n.engine";
            is_yolov8 = true;
            
            std::ifstream file(enginePath);
            if (!file.good()) {
                enginePath = modelPath + "/Yolov11ng3.engine";
                is_yolov8 = false;
                file = std::ifstream(enginePath);
                if (!file.good()) {
                    throw std::runtime_error("No TensorRT engine found. Expected yolov8n.engine or yolov7-tiny.engine");
                }
            }
            
            LOG("Loading TensorRT engine: " << enginePath);
            auto engineData = loadEngine(enginePath);
            
            runtime = nvinfer1::createInferRuntime(gLogger);
            if (!runtime) throw std::runtime_error("Failed to create TensorRT runtime");
            
            engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
            if (!engine) throw std::runtime_error("Failed to deserialize TensorRT engine");
            
            context = engine->createExecutionContext();
            if (!context) throw std::runtime_error("Failed to create TensorRT execution context");
            
            // Update deprecated binding APIs
            const int numBindings = engine->getNbBindings();
            inputBuffers.clear();
            outputBuffers.clear();
            outputBindingIndices.clear();
            outputSizes.clear();
            inputBindingIndex = -1;
    
            for (int i = 0; i < numBindings; i++) {
                const std::string bindingName = engine->getBindingName(i);
                nvinfer1::Dims dims = engine->getBindingDimensions(i);
                size_t size = 1;
                for (int j = 0; j < dims.nbDims; j++) {
                    size *= dims.d[j];
                }
                size *= sizeof(float);
    
                if (engine->bindingIsInput(i)) {
                    inputBindingIndex = i;
                    CudaBuffer inBuffer;
                    inBuffer.allocate(size);
                    inputBuffers.push_back(inBuffer);
                    width = dims.d[2];
                    height = dims.d[3];
                    LOG("Input binding: " << bindingName << ", index: " << i << ", shape: [" << 
                        dims.d[0] << "," << dims.d[1] << "," << width << "," << height << "], ptr: " << inBuffer.ptr);
                } else {
                    outputBindingIndices.push_back(i);
                    outputSizes.push_back(size);
                    CudaBuffer outBuffer;
                    outBuffer.allocate(size);
                    outputBuffers.push_back(outBuffer);
                    LOG("Output binding: " << bindingName << ", index: " << i << ", size: " << size << ", ptr: " << outBuffer.ptr);
                }
            }
    
            // Check for pointer overlaps
            std::set<void*> pointers;
            if (!inputBuffers.empty()) {
                pointers.insert(inputBuffers[0].ptr);
            }
            for (const auto& buf : outputBuffers) {
                if (pointers.count(buf.ptr) > 0) {
                    throw std::runtime_error("Buffer pointer overlap detected: " + 
                                           std::to_string(reinterpret_cast<uintptr_t>(buf.ptr)));
                }
                pointers.insert(buf.ptr);
            }
    
            initialized = true;
            LOG("TensorRT detector initialized successfully");
        } catch (const std::exception& e) {
            ERROR("Failed to initialize TensorRT detector: " << e.what());
            initialized = false;
        }
    }
    
    DetectionOutput runInference(const Mat& input) override {
        DetectionOutput output;
        
        try {
            // Validate input
            if (input.cols != width || input.rows != height) {
                throw std::runtime_error("Input size (" + std::to_string(input.cols) + "x" + 
                                       std::to_string(input.rows) + ") mismatches engine (" + 
                                       std::to_string(width) + "x" + std::to_string(height) + ")");
            }
    
            // Prepare input data
            Mat inputBlob;
            input.convertTo(inputBlob, CV_32FC3, 1.0f / 255.0f);
            std::vector<float> inputData(width * height * 3);
            float* inputDataPtr = inputData.data();
            for (int c = 0; c < 3; c++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        inputDataPtr[c * width * height + h * width + w] = inputBlob.at<cv::Vec3f>(h, w)[c];
                    }
                }
            }
    
            // Validate resources
            size_t inputSize = width * height * 3 * sizeof(float);
            if (inputBuffers.empty() || inputBuffers[0].ptr == nullptr) {
                throw std::runtime_error("Input buffer not allocated");
            }
            if (inputBuffers[0].size < inputSize) {
                throw std::runtime_error("Input buffer size too small");
            }
            if (inputDataPtr == nullptr) {
                throw std::runtime_error("Input data pointer is null");
            }
    
            LOG("Copying input: src=" << inputDataPtr << ", dst=" << inputBuffers[0].ptr << ", size=" << inputSize);
            cudaError_t err = cudaMemcpy(inputBuffers[0].ptr, inputDataPtr, inputSize, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
            }
            LOG("Copy succeeded");
    
            std::vector<void*> bindings(engine->getNbBindings(), nullptr);
            bindings[inputBindingIndex] = inputBuffers[0].ptr;
            for (size_t i = 0; i < outputBindingIndices.size(); ++i) {
                bindings[outputBindingIndices[i]] = outputBuffers[i].ptr;
            }
            for (int i = 0; i < bindings.size(); ++i) {
                LOG("Binding[" << i << "] = " << bindings[i]);
            }
    
            if (!context->enqueueV2(bindings.data(), stream, nullptr)) {
                cudaError_t cudaErr = cudaGetLastError();
                throw std::runtime_error("TensorRT enqueueV2 failed: " + std::string(cudaGetErrorString(cudaErr)));
            }
    
            err = cudaStreamSynchronize(stream);
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaStreamSynchronize failed: " + std::string(cudaGetErrorString(err)));
            }
    
            for (size_t i = 0; i < outputBuffers.size(); ++i) {
                std::vector<float> outputData(outputSizes[i] / sizeof(float));
                err = cudaMemcpy(outputData.data(), outputBuffers[i].ptr, outputSizes[i], cudaMemcpyDeviceToHost);
                if (err != cudaSuccess) {
                    throw std::runtime_error("cudaMemcpy failed for output " + std::to_string(i) + ": " + 
                                           std::string(cudaGetErrorString(err)));
                }
                // Reshape to [8400, 84] for YOLOv8
                int numDetections = 8400;
                int numElements = 84;  // 4 box coords + 1 conf + 79 class scores (assuming 80 classes)
                Mat reshaped(numDetections, numElements, CV_32F, outputData.data());
                output.outputs.push_back(reshaped.clone());
                LOG("Output " << i << " reshaped to: " << reshaped.rows << "x" << reshaped.cols);
                
                // Log sample data
                std::ostringstream oss;
                for (int j = 0; j < std::min(5, numDetections); ++j) {
                    oss << "Det[" << j << "]: x=" << reshaped.at<float>(j, 0) 
                        << ", y=" << reshaped.at<float>(j, 1) 
                        << ", w=" << reshaped.at<float>(j, 2) 
                        << ", h=" << reshaped.at<float>(j, 3) 
                        << ", conf=" << reshaped.at<float>(j, 4) << " ";
                }
                LOG("Sample detections: " << oss.str());
            }
    
            output.num_outputs = outputBuffers.size();
    
        } catch (const std::exception& e) {
            ERROR("Inference failed: " << e.what());
            cudaError_t cudaErr = cudaGetLastError();
            if (cudaErr != cudaSuccess) {
                ERROR("CUDA error: " << cudaGetErrorString(cudaErr));
            }
        }
        
        return output;
    }

    void processDetections(const DetectionOutput& output, Mat& frame, float scale, int dx, int dy) override {
        const int img_width = frame.cols;
        const int img_height = frame.rows;
        const float model_width = 640.0f;
        const float model_height = 640.0f;
    
        LOG("BOX_THRESH: " << BOX_THRESH);
    
        for (const auto& out : output.outputs) {
            const float* data = (float*)out.data;
            const int dimensions = out.cols;
            LOG("Processing output: " << out.rows << " rows, " << dimensions << " cols");
    
            #pragma omp parallel for schedule(dynamic, 16)
            for (int j = 0; j < out.rows; ++j) {
                const float confidence = data[j * dimensions + 4];
                if (confidence > BOX_THRESH && confidence <= 1.0f) {  // Ensure conf is valid
                    const Mat scores = out.row(j).colRange(5, dimensions);
                    Point classIdPoint;
                    double maxScore;
                    minMaxLoc(scores, 0, &maxScore, 0, &classIdPoint);
    
                    const float totalConfidence = confidence * maxScore;
                    if (totalConfidence > BOX_THRESH && classIdPoint.x == 0) {  // Only person (class 0)
                        // Normalized coords (0-1) to model size (640x640)
                        float centerX = data[j * dimensions + 0] * model_width;
                        float centerY = data[j * dimensions + 1] * model_height;
                        float w = data[j * dimensions + 2] * model_width;
                        float h = data[j * dimensions + 3] * model_height;
    
                        // Adjust to original frame size
                        int left = static_cast<int>((centerX - w / 2 - dx) / scale);
                        int top = static_cast<int>((centerY - h / 2 - dy) / scale);
                        int boxWidth = static_cast<int>(w / scale);
                        int boxHeight = static_cast<int>(h / scale);
    
                        // Clamp to frame boundaries
                        left = std::max(0, std::min(left, img_width - 1));
                        top = std::max(0, std::min(top, img_height - 1));
                        boxWidth = std::min(boxWidth, img_width - left);
                        boxHeight = std::min(boxHeight, img_height - top);
    
                        if (boxWidth > 0 && boxHeight > 0) {
                            #pragma omp critical
                            {
                                boxes.push_back(Rect(left, top, boxWidth, boxHeight));
                                confidences.push_back(totalConfidence);
                                classIds.push_back(classIdPoint.x);
                                LOG("Detected person: x=" << left << ", y=" << top << ", w=" << boxWidth 
                                    << ", h=" << boxHeight << ", conf=" << totalConfidence);
                            }
                        }
                    }
                }
            }
        }
    
        // Draw boxes
        for (size_t i = 0; i < boxes.size(); ++i) {
            rectangle(frame, boxes[i], Scalar(0, 255, 0), 2);
            putText(frame, "Person: " + std::to_string(confidences[i]), 
                    boxes[i].tl(), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
        }
        LOG("Drawn " << boxes.size() << " person boxes");
    }

    void releaseOutputs(const DetectionOutput& output) override {
        // Clear all data in the DetectionOutput struct
        output.clear();
    }
};

Detector* createDetector(const string& modelPath, const vector<string>& targetClasses) {
    try {
        return new TensorRTDetector(modelPath, targetClasses);
    } catch (const exception& e) {
        ERROR("Error creating TensorRT detector: " << e.what());
        return nullptr;
    }
}
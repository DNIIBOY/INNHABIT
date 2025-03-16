#include "detector.h"
#include "common.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <memory>
#include <algorithm>
#include <iostream>
#include <cmath>  // For std::exp

#define LOG(msg) std::cout << "[DEBUG] " << msg << std::endl
#define ERROR(msg) std::cerr << "[ERROR] " << msg << std::endl

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            switch (severity) {
                case Severity::kINTERNAL_ERROR: std::cerr << "[TRT ERROR] "; break;
                case Severity::kERROR: std::cerr << "[TRT ERROR] "; break;
                case Severity::kWARNING: std::cerr << "[TRT WARNING] "; break;
                case Severity::kINFO: std::cerr << "[TRT INFO] "; break;
                default: std::cerr << "[TRT UNKNOWN] "; break;
            }
            std::cerr << msg << std::endl;
        }
    }
};

struct TRTDeleter {
    template <typename T>
    void operator()(T* obj) const {
        if (obj) obj->destroy();
    }
};

template <typename T>
using TRTUniquePtr = std::unique_ptr<T, TRTDeleter>;

class TensorRTDetector : public GenericDetector {
private:
    static Logger gLogger;
    
    TRTUniquePtr<nvinfer1::IRuntime> runtime;
    TRTUniquePtr<nvinfer1::ICudaEngine> engine;
    TRTUniquePtr<nvinfer1::IExecutionContext> context;
    
    struct BufferManager {
        std::vector<void*> deviceBuffers;
        std::vector<void*> hostBuffers;
        std::vector<size_t> sizes;
        cudaStream_t stream;
        
        BufferManager() : stream(nullptr) {}
        
        ~BufferManager() {
            for (void* buffer : deviceBuffers) {
                if (buffer) cudaFree(buffer);
            }
            for (void* buffer : hostBuffers) {
                if (buffer) cudaFreeHost(buffer);
            }
            if (stream) cudaStreamDestroy(stream);
        }
    } bufferManager;
    
    float confidenceThreshold;
    float iouThreshold;
    int numClasses;
    int numBoxes;
    std::string inputName;
    std::string outputName;

    void checkCudaStatus(const char* operation) {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            ERROR(operation << " failed: " << cudaGetErrorString(err));
        }
    }

    void printDetections(const std::vector<Detection>& detections) const {
        for (size_t i = 0; i < std::min<size_t>(detections.size(), 3); ++i) {  // Print up to 3 detections
            const Detection& det = detections[i];
            std::cout << "  Detection " << i + 1 << ":" << std::endl;
            std::cout << "    Class: " << det.classId << std::endl;
            std::cout << "    Confidence: " << det.confidence << std::endl;
            std::cout << "    Bounding Box: [x=" << det.box.x << ", y=" << det.box.y
                      << ", width=" << det.box.width << ", height=" << det.box.height << "]" << std::endl;
        }
        if (detections.size() > 3) {
            LOG("... and " << (detections.size() - 3) << " more detections");
        }
    }
    
    bool buildEngine(const std::string& onnxFilePath, const std::string& engineFilePath) {
        LOG("Starting engine build process");
        LOG("ONNX path: " << onnxFilePath);
        LOG("Engine path: " << engineFilePath);
        
        std::ifstream engineFile(engineFilePath, std::ios::binary);
        if (engineFile.good()) {
            LOG("Found existing engine file");
            engineFile.seekg(0, std::ios::end);
            size_t size = engineFile.tellg();
            engineFile.seekg(0, std::ios::beg);
            std::vector<char> engineData(size);
            engineFile.read(engineData.data(), size);
            
            runtime.reset(nvinfer1::createInferRuntime(gLogger));
            if (!runtime) {
                ERROR("Runtime creation failed - check CUDA installation");
                return false;
            }
            engine.reset(runtime->deserializeCudaEngine(engineData.data(), size));
            if (!engine) {
                ERROR("Engine deserialization failed - possibly corrupted file or wrong TensorRT version");
                return false;
            }
            LOG("Successfully loaded existing engine");
            return true;
        }
        
        LOG("Building new engine from ONNX");
        TRTUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(gLogger));
        if (!builder) {
            ERROR("Builder creation failed - check TensorRT installation");
            return false;
        }
        LOG("Builder created successfully");
        
        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        TRTUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(explicitBatch));
        if (!network) {
            ERROR("Failed to create TensorRT network definition");
            return false;
        }
        LOG("Network definition created");
        
        TRTUniquePtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, gLogger));
        if (!parser) {
            ERROR("Failed to create ONNX parser");
            return false;
        }
        LOG("ONNX parser created");
        
        std::ifstream onnxFile(onnxFilePath, std::ios::binary);
        if (!onnxFile.good()) {
            ERROR("Failed to open ONNX file: " << onnxFilePath);
            return false;
        }
        onnxFile.seekg(0, std::ios::end);
        size_t size = onnxFile.tellg();
        onnxFile.seekg(0, std::ios::beg);
        std::vector<char> onnxData(size);
        onnxFile.read(onnxData.data(), size);
        
        if (!parser->parse(onnxData.data(), size)) {
            ERROR("Failed to parse ONNX model");
            return false;
        }
        LOG("ONNX model parsed successfully");
        
        TRTUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
        if (!config) {
            ERROR("Failed to create builder config");
            return false;
        }
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
        
        TRTUniquePtr<nvinfer1::IHostMemory> serializedEngine(builder->buildSerializedNetwork(*network, *config));
        if (!serializedEngine) {
            ERROR("Failed to build serialized engine");
            return false;
        }
        LOG("Engine built successfully");
        
        std::ofstream engineOutput(engineFilePath, std::ios::binary);
        if (!engineOutput.good()) {
            ERROR("Failed to open engine file for writing: " << engineFilePath);
            return false;
        }
        engineOutput.write(static_cast<const char*>(serializedEngine->data()), serializedEngine->size());
        
        runtime.reset(nvinfer1::createInferRuntime(gLogger));
        if (!runtime) {
            ERROR("Failed to create TensorRT runtime");
            return false;
        }
        engine.reset(runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size()));
        if (!engine) {
            ERROR("Failed to deserialize CUDA engine");
            return false;
        }
        LOG("Engine deserialized successfully");
        return true;
    }
    
    bool initializeBuffers() {
        LOG("Initializing buffers");
        if (cudaStreamCreate(&bufferManager.stream) != cudaSuccess) {
            ERROR("CUDA stream creation failed - check CUDA runtime");
            return false;
        }
        LOG("CUDA stream created");
        
        int numBindings = engine->getNbBindings();
        LOG("Number of bindings: " << numBindings);
        if (numBindings < 2) {
            ERROR("Invalid number of bindings - expected at least 1 input and 1 output");
        }
        
        bufferManager.deviceBuffers.resize(numBindings);
        bufferManager.hostBuffers.resize(numBindings);
        bufferManager.sizes.resize(numBindings);
        
        for (int i = 0; i < numBindings; i++) {
            nvinfer1::Dims dims = engine->getBindingDimensions(i);
            nvinfer1::DataType dtype = engine->getBindingDataType(i);
            LOG("Binding " << i << ": " << engine->getBindingName(i));
            LOG("  Dimensions: " << dims.nbDims);
            for (int j = 0; j < dims.nbDims; j++) {
                LOG("  Dim[" << j << "]: " << dims.d[j]);
            }
            
            size_t size = 1;
            for (int j = 0; j < dims.nbDims; j++) {
                size *= dims.d[j];
            }
            switch (dtype) {
                case nvinfer1::DataType::kFLOAT: size *= sizeof(float); break;
                case nvinfer1::DataType::kHALF: size *= sizeof(int16_t); break;
                case nvinfer1::DataType::kINT8: size *= sizeof(int8_t); break;
                case nvinfer1::DataType::kINT32: size *= sizeof(int32_t); break;
                default: ERROR("Unsupported data type"); return false;
            }
            bufferManager.sizes[i] = size;
            LOG("  Size in bytes: " << size);
            
            if (cudaMalloc(&bufferManager.deviceBuffers[i], size) != cudaSuccess) {
                ERROR("Failed to allocate device memory for binding " << i);
                return false;
            }
            if (cudaMallocHost(&bufferManager.hostBuffers[i], size) != cudaSuccess) {
                ERROR("Failed to allocate host memory for binding " << i);
                return false;
            }
            
            if (engine->bindingIsInput(i)) {
                inputName = engine->getBindingName(i);
                LOG("Input binding: " << inputName << ", size: " << size << " bytes");
            } else {
                outputName = engine->getBindingName(i);
                LOG("Output binding: " << outputName << ", size: " << size << " bytes");
                nvinfer1::Dims outputDims = engine->getBindingDimensions(i);
                if (outputDims.nbDims == 3) {
                    numBoxes = outputDims.d[2];
                }
            }
        }
        return true;
    }
    
    void preprocessImage(const cv::Mat& frame, float* inputBuffer) {
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(width, height));
        cv::Mat floatImage;
        resized.convertTo(floatImage, CV_32FC3, 1.0/255.0);
        cv::Mat rgbImage;
        cv::cvtColor(floatImage, rgbImage, cv::COLOR_BGR2RGB);
        std::vector<cv::Mat> channels(3);
        cv::split(rgbImage, channels);
        int channelSize = width * height;
        for (int c = 0; c < 3; c++) {
            memcpy(inputBuffer + c * channelSize, channels[c].data, channelSize * sizeof(float));
        }
    }
    
    void postprocessDetections(const float* outputBuffer, cv::Mat& frame, std::vector<Detection>& detections) {
        const int frameWidth = frame.cols;
        const int frameHeight = frame.rows;
        const float widthScale = static_cast<float>(frameWidth) / 640;
        const float heightScale = static_cast<float>(frameHeight) / 640;
        const int numBboxes = 8400;
        const int numElements = 5;  // x, y, w, h, score
        
        // Transpose on CPU: (5, 8400) -> (8400, 5)
        std::vector<float> transposed(numBboxes * numElements);
        for (int i = 0; i < numBboxes; i++) {
            for (int j = 0; j < numElements; j++) {
                transposed[i * numElements + j] = outputBuffer[j * numBboxes + i];
            }
        }
    
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> classIds;
        float minConf = 1.0f, maxConf = 0.0f;
        int aboveThreshold = 0;
    
        for (int i = 0; i < numBboxes; i++) {
            float centerX = transposed[i * numElements];
            float centerY = transposed[i * numElements + 1];
            float boxWidth = transposed[i * numElements + 2];
            float boxHeight = transposed[i * numElements + 3];
            float rawScore = transposed[i * numElements + 4];
            float confidence = rawScore;  // Test direct value
            // float confidence = 1.0f / (1.0f + std::exp(-rawScore));  // Uncomment to test sigmoid
            
            confidence = std::min(1.0f, std::max(0.0f, confidence));
            minConf = std::min(minConf, confidence);
            maxConf = std::max(maxConf, confidence);
            
            if (confidence > confidenceThreshold) {
                aboveThreshold++;
                float scaledCenterX = centerX * widthScale;
                float scaledCenterY = centerY * heightScale;
                float scaledWidth = boxWidth * widthScale;
                float scaledHeight = boxHeight * heightScale;
                
                int x = static_cast<int>(scaledCenterX - (scaledWidth / 2));
                int y = static_cast<int>(scaledCenterY - (scaledHeight / 2));
                int w = static_cast<int>(scaledWidth);
                int h = static_cast<int>(scaledHeight);
                
                w = std::max(w, 10);
                h = std::max(h, 10);
                x = std::max(0, std::min(x, frameWidth - 1));
                y = std::max(0, std::min(y, frameHeight - 1));
                w = std::min(w, frameWidth - x);
                h = std::min(h, frameHeight - y);
                
                boxes.push_back(cv::Rect(x, y, w, h));
                confidences.push_back(confidence);
                classIds.push_back(0);
            }
        }
        
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, confidenceThreshold, iouThreshold, indices);
        
        detections.clear();
        for (int idx : indices) {
            Detection det;
            det.classId = classIds[idx];
            det.confidence = confidences[idx];
            det.box = boxes[idx];
            detections.push_back(det);
        }
    }
public:
    TensorRTDetector(const std::string& modelPath, const std::vector<std::string>& targetClasses_)
        : GenericDetector(modelPath, targetClasses_), confidenceThreshold(0.1f), iouThreshold(0.1f), numClasses(1), numBoxes(0) {
        LOG("Initializing TensorRTDetector");
        std::string onnxFilePath = modelPath + "/Yolov11ng3.onnx";
        std::string engineFilePath = modelPath + "/Yolov11ng3.trt";

        if (!buildEngine(onnxFilePath, engineFilePath)) {
            ERROR("Failed to build TensorRT engine");
            initialized = false;
            return;
        }

        if (!initializeBuffers()) {
            ERROR("Failed to initialize TensorRT buffers");
            initialized = false;
            return;
        }

        context.reset(engine->createExecutionContext());
        if (!context) {
            ERROR("Failed to create TensorRT execution context");
            initialized = false;
            return;
        }
        LOG("Detector initialized successfully");
        initialized = true;
    }

    ~TensorRTDetector() {
        context.reset();
        engine.reset();
        runtime.reset();
    }

    void detect(cv::Mat& frame) override {
        if (!initialized) {
            ERROR("Detector not properly initialized");
            return;
        }

        detections.clear();
        float* inputBuffer = static_cast<float*>(bufferManager.hostBuffers[0]);
        preprocessImage(frame, inputBuffer);
        
        if (cudaMemcpyAsync(bufferManager.deviceBuffers[0], bufferManager.hostBuffers[0], 
            bufferManager.sizes[0], cudaMemcpyHostToDevice, bufferManager.stream) != cudaSuccess) {
            ERROR("Failed to copy input data to device");
            checkCudaStatus("cudaMemcpyAsync input");
            return;
        }
        
        if (!context->enqueueV2(bufferManager.deviceBuffers.data(), bufferManager.stream, nullptr)) {
            ERROR("Failed to execute TensorRT inference");
            return;
        }

        // Use correct output binding (index 4 for 'outputs')
        int outputIndex = 4;  // Binding 4 is 'outputs'
        float* outputBuffer = static_cast<float*>(bufferManager.hostBuffers[outputIndex]);
        if (cudaMemcpyAsync(bufferManager.hostBuffers[outputIndex], bufferManager.deviceBuffers[outputIndex], 
            bufferManager.sizes[outputIndex], cudaMemcpyDeviceToHost, bufferManager.stream) != cudaSuccess) {
            ERROR("Failed to copy output data to host");
            checkCudaStatus("cudaMemcpyAsync output");
            return;
        }
        
        cudaStreamSynchronize(bufferManager.stream);
        checkCudaStatus("cudaStreamSynchronize");
        
        postprocessDetections(outputBuffer, frame, detections);
        drawDetections(frame, detections);
    }

protected:
    void initialize(const std::string& modelPath) override {}
    DetectionOutput runInference(const cv::Mat& input) override { return DetectionOutput(); }
    void releaseOutputs(const DetectionOutput& output) override {}
};

Logger TensorRTDetector::gLogger;

Detector* createDetector(const std::string& modelPath, const std::vector<std::string>& targetClasses) {
    try {
        LOG("Creating new TensorRT detector");
        return new TensorRTDetector(modelPath, targetClasses);
    } catch (const std::exception& e) {
        ERROR("Error creating TensorRT detector: " << e.what());
        return nullptr;
    }
}
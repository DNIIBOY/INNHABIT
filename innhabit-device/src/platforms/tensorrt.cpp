#include "detector.h"
#include "common.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>  // Added for FP16 support
#include <fstream>
#include <memory>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

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

    void checkCudaStatus(cudaError_t status, const char* operation) {
        if (status != cudaSuccess) {
            ERROR(operation << " failed: " << cudaGetErrorString(status));
        }
    }

    void printDetections(const std::vector<Detection>& detections) const {
        for (size_t i = 0; i < std::min<size_t>(detections.size(), 3); ++i) {
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
            ERROR("Engine deserialization failed");
            return false;
        }
        LOG("Successfully loaded existing engine");
        for (int i = 0; i < engine->getNbBindings(); i++) {
            LOG("Binding " << i << " data type: " << static_cast<int>(engine->getBindingDataType(i)));
        }
        return true;
    }
    
    LOG("Building new engine from ONNX");
    TRTUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(gLogger));
    if (!builder) {
        ERROR("Builder creation failed - check TensorRT installation");
        return false;
    }
    
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TRTUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(explicitBatch));
    if (!network) {
        ERROR("Failed to create TensorRT network definition");
        return false;
    }
    
    TRTUniquePtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, gLogger));
    if (!parser) {
        ERROR("Failed to create ONNX parser");
        return false;
    }
    
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
    
    TRTUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) {
        ERROR("Failed to create builder config");
        return false;
    }
    config->setMaxWorkspaceSize(1ULL << 30);  // 1GB workspace
    config->setFlag(nvinfer1::BuilderFlag::kFP16);  // Enable FP16 precision
    config->clearFlag(nvinfer1::BuilderFlag::kTF32);  // Disable TensorFloat-32
    if (!builder->platformHasFastFp16()) {
        ERROR("Platform does not support fast FP16 - falling back to FP32 might occur");
    }

    TRTUniquePtr<nvinfer1::IHostMemory> serializedEngine(builder->buildSerializedNetwork(*network, *config));
    if (!serializedEngine) {
        ERROR("Failed to build serialized engine");
        return false;
    }
    
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
    for (int i = 0; i < engine->getNbBindings(); i++) {
        LOG("Binding " << i << " data type: " << static_cast<int>(engine->getBindingDataType(i)));
    }
    return true;
}
    
    bool initializeBuffers() {
        LOG("Initializing buffers");
        if (cudaStreamCreate(&bufferManager.stream) != cudaSuccess) {
            ERROR("CUDA stream creation failed - check CUDA runtime");
            return false;
        }
        
        int numBindings = engine->getNbBindings();
        LOG("Number of bindings: " << numBindings);
        if (numBindings < 2) {
            ERROR("Invalid number of bindings - expected at least 1 input and 1 output");
            return false;
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
            if (dtype == nvinfer1::DataType::kHALF) {
                size *= sizeof(__half);  // FP16
                LOG("  Data type: FP16");
            } else if (dtype == nvinfer1::DataType::kFLOAT) {
                size *= sizeof(float);  // FP32
                LOG("  Data type: FP32");
            } else {
                ERROR("Unsupported data type: " << static_cast<int>(dtype));
                return false;
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
    
    void preprocess(const cv::Mat& frame, void* inputBuffer) {  // Use void* to handle both FP16 and FP32
        const cv::Scalar PADDING_COLOR(114, 114, 114);
        cv::cuda::GpuMat gpu_frame, gpu_rgb, gpu_resized_part, gpu_output;
        cv::cuda::Stream cvStream;

        gpu_frame.upload(frame, cvStream);
        cv::cuda::cvtColor(gpu_frame, gpu_rgb, cv::COLOR_BGR2RGB, 0, cvStream);

        const int img_width = frame.cols;
        const int img_height = frame.rows;
        float scale = std::min(static_cast<float>(width) / img_width, static_cast<float>(height) / img_height);
        const int new_width = static_cast<int>(img_width * scale);
        const int new_height = static_cast<int>(img_height * scale);
        const int dx = (width - new_width) / 2;
        const int dy = (height - new_height) / 2;

        cv::cuda::resize(gpu_rgb, gpu_resized_part, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR, cvStream);
        gpu_output.create(height, width, CV_8UC3);
        gpu_output.setTo(PADDING_COLOR, cvStream);
        gpu_resized_part.copyTo(gpu_output(cv::Rect(dx, dy, new_width, new_height)), cvStream);

        // Convert based on engine data type
        cv::Mat cpu_float;
        const int channelSize = width * height;
        nvinfer1::DataType dtype = engine->getBindingDataType(0);  // Assuming binding 0 is input
        if (dtype == nvinfer1::DataType::kHALF) {
            cv::cuda::GpuMat gpu_half;
            gpu_output.convertTo(gpu_half, CV_16F, 1.0 / 255.0, 0, cvStream);
            gpu_half.download(cpu_float, cvStream);
            cvStream.waitForCompletion();
            std::vector<cv::Mat> channels(3);
            cv::split(cpu_float, channels);
            __half* ptr = static_cast<__half*>(inputBuffer);
            for (int i = 0; i < 3; ++i) {
                std::vector<__half> channelData(channelSize);
                for (int j = 0; j < channelSize; ++j) {
                    channelData[j] = __float2half(channels[i].at<float>(j));
                }
                checkCudaStatus(cudaMemcpyAsync(ptr + i * channelSize, channelData.data(),
                                                channelSize * sizeof(__half), cudaMemcpyHostToDevice,
                                                bufferManager.stream), "Channel upload");
            }
        } else {  // FP32
            cv::cuda::GpuMat gpu_float;
            gpu_output.convertTo(gpu_float, CV_32F, 1.0 / 255.0, 0, cvStream);
            gpu_float.download(cpu_float, cvStream);
            cvStream.waitForCompletion();
            std::vector<cv::Mat> channels(3);
            cv::split(cpu_float, channels);
            float* ptr = static_cast<float*>(inputBuffer);
            for (int i = 0; i < 3; ++i) {
                checkCudaStatus(cudaMemcpyAsync(ptr + i * channelSize, channels[i].data,
                                                channelSize * sizeof(float), cudaMemcpyHostToDevice,
                                                bufferManager.stream), "Channel upload");
            }
        }

        checkCudaStatus(cudaStreamSynchronize(bufferManager.stream), "Stream sync in preprocess");
    }
    
    void postprocess(const void* outputBuffer, cv::Mat& frame, std::vector<Detection>& detections) {
        const int frameWidth = frame.cols;
        const int frameHeight = frame.rows;
        const float widthScale = static_cast<float>(frameWidth) / 640;
        const float heightScale = static_cast<float>(frameHeight) / 640;
        const int numBboxes = 8400;
        const int numElements = 5;

        LOG("Postprocessing - Frame size: " << frameWidth << "x" << frameHeight);
        if (!outputBuffer) {
            ERROR("Output buffer is null");
            return;
        }

        std::vector<float> transposed(numBboxes * numElements);
        nvinfer1::DataType dtype = engine->getBindingDataType(4);  // Assuming binding 4 is 'outputs'
        if (dtype == nvinfer1::DataType::kHALF) {
            const __half* halfBuffer = static_cast<const __half*>(outputBuffer);
            for (int i = 0; i < numBboxes; i++) {
                for (int j = 0; j < numElements; j++) {
                    transposed[i * numElements + j] = __half2float(halfBuffer[j * numBboxes + i]);
                }
            }
        } else {  // FP32
            const float* floatBuffer = static_cast<const float*>(outputBuffer);
            for (int i = 0; i < numBboxes; i++) {
                for (int j = 0; j < numElements; j++) {
                    transposed[i * numElements + j] = floatBuffer[j * numBboxes + i];
                }
            }
        }

        // Rest of the postprocessing logic remains unchanged
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> classIds;

        int validDetections = 0;
        for (int i = 0; i < numBboxes; i++) {
            float centerX = transposed[i * numElements];
            float centerY = transposed[i * numElements + 1];
            float boxWidth = transposed[i * numElements + 2];
            float boxHeight = transposed[i * numElements + 3];
            float confidence = transposed[i * numElements + 4];

            if (confidence > confidenceThreshold) {
                validDetections++;
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
                x = std::max(0, std::min(x, frameWidth - w));
                y = std::max(0, std::min(y, frameHeight - h));

                boxes.push_back(cv::Rect(x, y, w, h));
                confidences.push_back(confidence);
                classIds.push_back(0);
            }
        }

        LOG("Found " << validDetections << " boxes above threshold " << confidenceThreshold);

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, NMS_THRESH, BOX_THRESH, indices);

        detections.clear();
        detections.reserve(indices.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            Detection det;
            det.classId = 0 <= classIds[idx] && classIds[idx] < numClasses ? targetClasses[classIds[idx]] : "unknown";
            det.confidence = confidences[idx];
            det.box = boxes[idx];
            detections.push_back(det);
        }

        LOG("Postprocess: " << detections.size() << " detections after NMS");
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
            ERROR("Detector not initialized");
            return;
        }

        if (frame.empty()) {
            ERROR("Input frame is empty");
            return;
        }

        detections.clear();
        if (bufferManager.hostBuffers.size() < 5) {
            ERROR("Insufficient host buffers: " << bufferManager.hostBuffers.size());
            return;
        }

        void* inputBuffer = bufferManager.hostBuffers[0];  // Changed to __half
        void* outputBuffer = bufferManager.hostBuffers[4];  // Changed to __half

        auto start = std::chrono::high_resolution_clock::now();

        auto t1 = std::chrono::high_resolution_clock::now();
        preprocess(frame, inputBuffer);
        auto t2 = std::chrono::high_resolution_clock::now();

        cudaEvent_t event, infStart, infStop;
        checkCudaStatus(cudaEventCreate(&event), "Event create");
        checkCudaStatus(cudaEventCreate(&infStart), "InfStart create");
        checkCudaStatus(cudaEventCreate(&infStop), "InfStop create");

        auto t3 = std::chrono::high_resolution_clock::now();
        checkCudaStatus(cudaMemcpyAsync(bufferManager.deviceBuffers[0], inputBuffer,
                bufferManager.sizes[0], cudaMemcpyHostToDevice, bufferManager.stream),
                "Input copy");
        auto t4 = std::chrono::high_resolution_clock::now();
        
        auto t5 = std::chrono::high_resolution_clock::now();
        checkCudaStatus(cudaEventRecord(infStart, bufferManager.stream), "InfStart record");
        if (!context->enqueueV2(bufferManager.deviceBuffers.data(), bufferManager.stream, nullptr)) {
            ERROR("Inference failed");
            cudaEventDestroy(event);
            cudaEventDestroy(infStart);
            cudaEventDestroy(infStop);
            return;
        }
        checkCudaStatus(cudaEventRecord(infStop, bufferManager.stream), "InfStop record");
        checkCudaStatus(cudaEventSynchronize(infStop), "InfStop sync");
        float inferenceMs;
        checkCudaStatus(cudaEventElapsedTime(&inferenceMs, infStart, infStop), "Elapsed time");
        auto t6 = std::chrono::high_resolution_clock::now();

        auto t7 = std::chrono::high_resolution_clock::now();
        checkCudaStatus(cudaMemcpyAsync(outputBuffer, bufferManager.deviceBuffers[4],
                bufferManager.sizes[4], cudaMemcpyDeviceToHost, bufferManager.stream),
                "Output copy");
        auto t8 = std::chrono::high_resolution_clock::now();

        auto t9 = std::chrono::high_resolution_clock::now();
        checkCudaStatus(cudaEventRecord(event, bufferManager.stream), "Event record");
        checkCudaStatus(cudaEventSynchronize(event), "Event sync");
        cudaEventDestroy(event);
        cudaEventDestroy(infStart);
        cudaEventDestroy(infStop);
        auto t10 = std::chrono::high_resolution_clock::now();

        auto t11 = std::chrono::high_resolution_clock::now();
        postprocess(outputBuffer, frame, detections);
        auto t12 = std::chrono::high_resolution_clock::now();

        auto t13 = std::chrono::high_resolution_clock::now();
        drawDetections(frame, detections);
        auto t14 = std::chrono::high_resolution_clock::now();

        auto end = std::chrono::high_resolution_clock::now();

        double preprocessTime = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double h2dTime = std::chrono::duration<double, std::milli>(t4 - t3).count();
        double inferenceTime = static_cast<double>(inferenceMs);
        double d2hTime = std::chrono::duration<double, std::milli>(t8 - t7).count();
        double syncTime = std::chrono::duration<double, std::milli>(t10 - t9).count();
        double postprocessTime = std::chrono::duration<double, std::milli>(t12 - t11).count();
        double drawTime = std::chrono::duration<double, std::milli>(t14 - t13).count();
        double totalTime = std::chrono::duration<double, std::milli>(end - start).count();

        LOG("Detect Timing (ms):");
        LOG("  Preprocess: " << preprocessTime);
        LOG("  H2D Copy: " << h2dTime);
        LOG("  Inference: " << inferenceTime);
        LOG("  D2H Copy: " << d2hTime);
        LOG("  Sync: " << syncTime);
        LOG("  Postprocess: " << postprocessTime);
        LOG("  Draw: " << drawTime);
        LOG("  Total: " << totalTime);
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
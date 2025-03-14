#include "detector.h"
#include "common.h"
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <memory>

using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
} gLogger;

class JetsonTensorRTDetector : public GenericDetector {
private:
    std::unique_ptr<IRuntime> runtime;
    std::unique_ptr<ICudaEngine> engine;
    std::unique_ptr<IExecutionContext> context;
    cudaStream_t stream;
    std::vector<void*> buffers;
    int inputIndex, outputIndex;

public:
    JetsonTensorRTDetector(const string& modelPath, const vector<string>& targetClasses_)
        : GenericDetector(modelPath, targetClasses_) {
        initialize(modelPath);
    }

    ~JetsonTensorRTDetector() {
        cudaStreamDestroy(stream);
        for (void* buf : buffers) {
            cudaFree(buf);
        }
    }

protected:
    void initialize(const string& modelPath) override {
        std::string engineFilePath = modelPath + "/Yolov11ng3.engine";
        std::ifstream engineFile(engineFilePath, std::ios::binary);
        if (!engineFile) {
            throw std::runtime_error("Failed to open engine file: " + engineFilePath);
        }

        engineFile.seekg(0, std::ios::end);
        size_t engineSize = engineFile.tellg();
        engineFile.seekg(0, std::ios::beg);
        std::vector<char> engineData(engineSize);
        engineFile.read(engineData.data(), engineSize);
        engineFile.close();

        runtime.reset(createInferRuntime(gLogger));
        engine.reset(runtime->deserializeCudaEngine(engineData.data(), engineSize));
        context.reset(engine->createExecutionContext());

        if (!engine || !context) {
            throw std::runtime_error("Failed to create TensorRT engine or execution context");
        }

        // Verify binding names and indices
        for (int i = 0; i < engine->getNbBindings(); ++i) {
            Dims dims = engine->getBindingDimensions(i);
            std::cout << "Binding " << i << " (" << engine->getBindingName(i) << "): ";
            for (int j = 0; j < dims.nbDims; ++j) {
                std::cout << dims.d[j] << " ";
            }
            std::cout << std::endl;
        }

        // Get binding indices by name
        inputIndex = engine->getBindingIndex("images");
        outputIndex = engine->getBindingIndex("outputs");

        // Allocate memory for all bindings
        buffers.resize(engine->getNbBindings());
        for (int i = 0; i < engine->getNbBindings(); ++i) {
            Dims dims = engine->getBindingDimensions(i);
            size_t size = 1;
            for (int j = 0; j < dims.nbDims; ++j) {
                size *= dims.d[j];
            }
            size *= sizeof(float);  // Assuming FP32 data type

            cudaError_t err = cudaMalloc(&buffers[i], size);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to allocate CUDA memory for binding " + std::to_string(i) + ": " + std::string(cudaGetErrorString(err)));
            }
        }

        cudaStreamCreate(&stream);

        initialized = true;
    }

    DetectionOutput runInference(const Mat& input) override {
        Mat blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, Size(width, height), Scalar(0, 0, 0), true, false);

        // Copy input data to GPU
        cudaError_t err = cudaMemcpyAsync(buffers[inputIndex], blob.ptr<float>(), width * height * channel * sizeof(float), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy input data to GPU: " + std::string(cudaGetErrorString(err)));
        }

        // Run inference
        if (!context->enqueueV2(buffers.data(), stream, nullptr)) {
            throw std::runtime_error("Failed to run inference");
        }

        // Copy output data from GPU
        DetectionOutput output;
        output.outputs.resize(1);

        // Get output dimensions
        Dims outputDims = engine->getBindingDimensions(outputIndex);
        size_t outputSize = 1;
        for (int i = 0; i < outputDims.nbDims; ++i) {
            outputSize *= outputDims.d[i];
        }

        // Allocate output buffer
        output.outputs[0].create(outputDims.d[1], outputDims.d[2], CV_32F);  // Adjust dimensions as needed

        err = cudaMemcpyAsync(output.outputs[0].ptr<float>(), buffers[outputIndex], outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy output data from GPU: " + std::string(cudaGetErrorString(err)));
        }

        // Synchronize the stream
        cudaStreamSynchronize(stream);

        output.num_outputs = 1;
        return output;
    }

    void processDetections(const DetectionOutput& output, Mat& frame, float scale, int dx, int dy) override {
        vector<Rect> boxes;
        vector<float> confidences;
        vector<int> classIds;

        int img_width = frame.cols;
        int img_height = frame.rows;

        // Process the correct output binding (e.g., binding index 4)
        const Mat& detectionsOutput = output.outputs[0];  // Index 0 corresponds to the first output binding

        for (int i = 0; i < detectionsOutput.rows; ++i) {
            const float* data = detectionsOutput.ptr<float>(i);  // Use const float*
            float confidence = data[4];
            if (confidence > BOX_THRESH) {
                Mat scores = detectionsOutput.row(i).colRange(5, detectionsOutput.cols);
                Point classIdPoint;
                double maxScore;
                minMaxLoc(scores, 0, &maxScore, 0, &classIdPoint);

                float totalConfidence = confidence * maxScore;
                if (totalConfidence > BOX_THRESH) {
                    // Normalized coordinates from the model
                    float centerX = data[0];  // Normalized center x
                    float centerY = data[1];  // Normalized center y
                    float w = data[2];       // Normalized width
                    float h = data[3];       // Normalized height

                    // Scale coordinates to the original image dimensions
                    int left = static_cast<int>((centerX - w / 2) * img_width);
                    int top = static_cast<int>((centerY - h / 2) * img_height);
                    int boxWidth = static_cast<int>(w * img_width);
                    int boxHeight = static_cast<int>(h * img_height);

                    // Clamp coordinates to ensure they are within the image boundaries
                    left = clamp(left, 0, img_width - 1);
                    top = clamp(top, 0, img_height - 1);
                    boxWidth = clamp(boxWidth, 0, img_width - left);
                    boxHeight = clamp(boxHeight, 0, img_height - top);

                    boxes.push_back(Rect(left, top, boxWidth, boxHeight));
                    confidences.push_back(totalConfidence);
                    classIds.push_back(classIdPoint.x);
                }
            }
        }

        vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, BOX_THRESH, NMS_THRESH, indices);

        detections.clear();
        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            string className = classIds[idx] < 80 ? COCO_LABELS[classIds[idx]] : "unknown";
            if (!targetClasses.empty() && find(targetClasses.begin(), targetClasses.end(), className) == targetClasses.end()) continue;

            Detection det;
            det.classId = className;
            det.confidence = confidences[idx];
            det.box = boxes[idx];
            detections.push_back(det);
        }
    }

    void releaseOutputs(const DetectionOutput& output) override {
        // No-op: TensorRT manages its own memory
    }
};

Detector* createDetector(const string& modelPath, const vector<string>& targetClasses) {
    try {
        return new JetsonTensorRTDetector(modelPath, targetClasses);
    } catch (const exception& e) {
        ERROR("Error creating Jetson TensorRT detector: " << e.what());
        return nullptr;
    }
}
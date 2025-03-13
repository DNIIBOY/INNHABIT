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
            std::cout << "Binding " << i << ": " << engine->getBindingName(i) << " ("
                      << (engine->bindingIsInput(i) ? "Input" : "Output")
                      << ")" << std::endl;
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
            cudaMalloc(&buffers[i], size);
        }

        cudaStreamCreate(&stream);

        initialized = true;
    }

    DetectionOutput runInference(const Mat& input) override {
        Mat blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, Size(width, height), Scalar(0, 0, 0), true, false);
        cudaMemcpyAsync(buffers[inputIndex], blob.ptr<float>(), width * height * channel * sizeof(float), cudaMemcpyHostToDevice, stream);

        context->enqueueV2(buffers.data(), stream, nullptr);

        DetectionOutput output;
        output.outputs.resize(1);
        output.outputs[0].create(height, width, CV_32F);
        cudaMemcpyAsync(output.outputs[0].ptr<float>(), buffers[outputIndex], width * height * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        output.num_outputs = 1;
        return output;
    }

    void processDetections(const DetectionOutput& output, Mat& frame, float scale, int dx, int dy) override {
        vector<Rect> boxes;
        vector<float> confidences;
        vector<int> classIds;

        for (int i = 0; i < output.outputs[0].rows; ++i) {
            const float* data = output.outputs[0].ptr<float>(i);  // Use const float*
            float confidence = data[4];
            if (confidence > BOX_THRESH) {
                Mat scores = output.outputs[0].row(i).colRange(5, output.outputs[0].cols);
                Point classIdPoint;
                double maxScore;
                minMaxLoc(scores, 0, &maxScore, 0, &classIdPoint);

                float totalConfidence = confidence * maxScore;
                if (totalConfidence > BOX_THRESH) {
                    float centerX = data[0] * width;
                    float centerY = data[1] * height;
                    float w = data[2] * width;
                    float h = data[3] * height;

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
#include "detector.h"
#include "common.h"
#include "logging.h"
#include "cuda_utils.h"
#include "macros.h"
#include "preprocess.h"
#include <NvOnnxParser.h>
#include "common.h"
#include <fstream>
#include <iostream>
#include "NvInfer.h"
#include <opencv2/opencv.hpp>

using namespace nvinfer1;
using namespace std;
using namespace cv;

#define isFP16 true
#define warmup true

Logger logger; // Use the logger from logging.h instead of redefining

class TensorRTDetector : public GenericDetector {
private:
        
    float* gpu_buffers[2];               //!< The vector of device buffers needed for engine execution
    float* cpu_output_buffer;

    cudaStream_t stream;
    IRuntime* runtime;                 //!< The TensorRT runtime used to deserialize the engine
    ICudaEngine* engine;               //!< The TensorRT engine used to run the network
    IExecutionContext* context;        //!< The context for executing inference using an ICudaEngine

    // Model parameters
    int model_input_w;
    int model_input_h;
    int num_detections;
    int detection_attribute_size;
    int num_classes = 1;
    const int MAX_IMAGE_SIZE = 2048 * 2048;
    float conf_threshold = BOX_THRESH;
    float nms_threshold = NMS_THRESH;
    
    
    void printDetections(const std::vector<Detection>& detections) const {
        size_t displayCount = std::min<size_t>(detections.size(), 3);
        for (size_t i = 0; i < displayCount; ++i) {
            const Detection& det = detections[i];
            std::cout << "  Detection " << i + 1 << ": Class=" << det.classId 
                      << ", Conf=" << det.confidence << ", Box=[x=" << det.box.x 
                      << ",y=" << det.box.y << ",w=" << det.box.width 
                      << ",h=" << det.box.height << "]" << std::endl;
        }
        if (detections.size() > displayCount) {
            std::cout << "... and " << (detections.size() - displayCount) << " more" << std::endl;
        }
    }
    
    bool buildEngine(const std::string& onnxPath, nvinfer1::ILogger& logger) {
        
        // create tensorrt builder
        auto builder = createInferBuilder(logger);
        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
        IBuilderConfig* config = builder->createBuilderConfig();
        if (isFP16)
        {
            config->setFlag(BuilderFlag::kFP16);
        }
        nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
        bool parsed = parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
        IHostMemory* plan{ builder->buildSerializedNetwork(*network, *config) };

        runtime = createInferRuntime(logger);

        engine = runtime->deserializeCudaEngine(plan->data(), plan->size());

        context = engine->createExecutionContext();

        delete network;
        delete config;
        delete parser;
        delete plan;
    }

    bool saveEngine(const std::string& onnxpath) {
        // Create an engine path from onnx path
        std::string engine_path;
        size_t dotIndex = onnxpath.find_last_of(".");
        if (dotIndex != std::string::npos) {
            engine_path = onnxpath.substr(0, dotIndex) + ".engine";
        }
        else
        {
            return false;
        }

        // Save the engine to the path
        if (engine)
        {
            nvinfer1::IHostMemory* data = engine->serialize();
            std::ofstream file;
            file.open(engine_path, std::ios::binary | std::ios::out);
            if (!file.is_open())
            {
                std::cout << "Create engine file" << engine_path << " failed" << std::endl;
                return 0;
            }
            file.write((const char*)data->data(), data->size());
            file.close();

            delete data;
        }
        return true;
    }
    
    void preprocess(cv::Mat& frame) {
        // Preprocessing data on gpu
        cuda_preprocess(frame.ptr(), frame.cols, frame.rows, gpu_buffers[0], model_input_w, model_input_h, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    
    void postprocess(std::vector<Detection>& output, cv::Mat& frame) {
        // Pre-allocate output vector to avoid dynamic resizing
        output.reserve(100); // Adjust based on expected max detections

        // Synchronize and copy data from GPU
        CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer, gpu_buffers[1], 
                                detection_attribute_size * num_detections * sizeof(float), 
                                cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Use a single pass to filter and scale boxes
        const Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);
        
        // Scaling factors (assuming these are class members or passed in)
        const float ratio_h = static_cast<float>(model_input_h) / frame.rows; // Video height: 1280x720
        const float ratio_w = static_cast<float>(model_input_w) / frame.cols; // Video width: 1280x720
        const float scale = std::min(ratio_h, ratio_w);
        const float offset_x = (ratio_h > ratio_w) ? 0.0f : (model_input_w - ratio_h * frame.cols) * 0.5f;
        const float offset_y = (ratio_h > ratio_w) ? (model_input_h - ratio_w * frame.rows) * 0.5f : 0.0f;

        // Temporary storage for NMS
        std::vector<Rect> boxes;
        std::vector<float> confidences;
        int class_Id;
        boxes.reserve(100); // Pre-allocate to avoid resizing
        confidences.reserve(100);

        // Single pass: filter and scale
        for (int i = 0; i < num_detections; ++i) {
            float confidence = det_output.at<float>(4, i);
            int class_Id = det_output.at<int>(5, i);
            if (confidence > conf_threshold) {
                float cx = det_output.at<float>(0, i);
                float cy = det_output.at<float>(1, i);
                float ow = det_output.at<float>(2, i);
                float oh = det_output.at<float>(3, i);

                // Scale and adjust box coordinates
                Rect box;
                box.x = static_cast<int>((cx - 0.5f * ow - offset_x) / scale);
                box.y = static_cast<int>((cy - 0.5f * oh - offset_y) / scale);
                box.width = static_cast<int>(ow / scale);
                box.height = static_cast<int>(oh / scale);

                boxes.push_back(box);
                confidences.push_back(confidence);
            }
        }

        // NMS
        std::vector<int> nms_result;
        dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_result);

        // Populate output directly
        for (int idx : nms_result) {
            Detection result;
            result.classId = COCO_LABELS[class_Id]; // Single-class (use 0 if classId is int)
            result.confidence = confidences[idx];
            result.box = boxes[idx];
            output.push_back(std::move(result)); // Move to avoid copying
        }
    }

public:
    TensorRTDetector(const std::string& model_path, const vector<string>& targetClasses)
            : GenericDetector(model_path, targetClasses) {
            if (model_path.find(".onnx") == std::string::npos) {
                initialize(model_path);
            } else {
                buildEngine(model_path, logger);
                saveEngine(model_path);
            }
        }

    ~TensorRTDetector() {
        // Memory free
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
        for (float*& buffer : gpu_buffers) {
            if (buffer) {
                cudaError_t err = cudaFree(buffer);
                if (err != cudaSuccess) {
                    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
                }
                buffer = nullptr;
            }
        }
        delete[] cpu_output_buffer;
        cuda_preprocess_destroy();
        delete context;
        delete engine;
        delete runtime;
    }

    void detect(cv::Mat& frame) override {
        if (!initialized) {
            std::cerr << "Detector not initialized" << std::endl;
            return;
        }

        // Clear previous detections
        detections.clear();

        // Total start time
        auto total_start = std::chrono::high_resolution_clock::now();

        // Preprocessing
        auto preprocess_start = std::chrono::high_resolution_clock::now();
        preprocess(frame);
        auto preprocess_end = std::chrono::high_resolution_clock::now();
        double preprocess_time = std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();

        // Inference
        auto inference_start = std::chrono::high_resolution_clock::now();
        runInference(frame);
        auto inference_end = std::chrono::high_resolution_clock::now();
        double inference_time = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();

        // Postprocessing
        auto postprocess_start = std::chrono::high_resolution_clock::now();
        postprocess(detections, frame);
        auto postprocess_end = std::chrono::high_resolution_clock::now();
        double postprocess_time = std::chrono::duration<double, std::milli>(postprocess_end - postprocess_start).count();
        // Total end time
        auto total_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();

        // Output timings
        std::cout << "Preprocessing time: " << preprocess_time << "ms" << std::endl;
        std::cout << "Inference time: " << inference_time << "ms" << std::endl;
        std::cout << "Postprocessing time: " << postprocess_time << "ms" << std::endl;
        std::cout << "Total detection time: " << total_time << "ms" << std::endl;
    }

    void draw(Mat& image, const vector<Detection>& output)
    {
        const float ratio_h = model_input_h / (float)image.rows;
        const float ratio_w = model_input_w / (float)image.cols;

        for (int i = 0; i < output.size(); i++)
        {
            auto detection = output[i];
            auto box = detection.box;
            auto class_id = detection.classId;
            auto conf = detection.confidence;
            cv::Scalar color = cv::Scalar(144, 144, 144);

            if (ratio_h > ratio_w)
            {
                box.x = box.x / ratio_w;
                box.y = (box.y - (model_input_h - ratio_w * image.rows) / 2) / ratio_w;
                box.width = box.width / ratio_w;
                box.height = box.height / ratio_w;
            }
            else
            {
                box.x = (box.x - (model_input_w - ratio_h * image.cols) / 2) / ratio_h;
                box.y = box.y / ratio_h;
                box.width = box.width / ratio_h;
                box.height = box.height / ratio_h;
            }

            rectangle(image, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), color, 3);

            // Detection box text
            string class_string = "person" + ' ' + to_string(conf).substr(0, 4);
            Size text_size = getTextSize(class_string, FONT_HERSHEY_DUPLEX, 1, 2, 0);
            Rect text_rect(box.x, box.y - 40, text_size.width + 10, text_size.height + 20);
            rectangle(image, text_rect, color, FILLED);
            putText(image, class_string, Point(box.x + 5, box.y - 10), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 2, 0);
    }
    }

    void initialize(const string& engineFilePath) override {
        ifstream engineStream(engineFilePath, ios::binary);
        engineStream.seekg(0, ios::end);
        const size_t modelSize = engineStream.tellg();
        engineStream.seekg(0, ios::beg);
        unique_ptr<char[]> engineData(new char[modelSize]);
        engineStream.read(engineData.get(), modelSize);
        engineStream.close();

        runtime = createInferRuntime(logger);
        engine = runtime->deserializeCudaEngine(engineData.get(), modelSize);
        context = engine->createExecutionContext();

        int numBindings = engine->getNbBindings();
        LOG("Number of bindings: " << numBindings);
        for (int i = 0; i < numBindings; i++) {
            Dims dims = engine->getBindingDimensions(i);
            std::string bindingName = engine->getBindingName(i);
            LOG("Binding " << i << ": " << bindingName << " [Dims: " << dims.d[0] << "x" << dims.d[1] << "x" << dims.d[2] << "x" << dims.d[3] << "]");
        }
        // Warning this only works with simplified models
        // Input binding (index 0) [1x3x640x640]
        model_input_h = engine->getBindingDimensions(0).d[2];
        model_input_w = engine->getBindingDimensions(0).d[3];
        LOG("[INFO] model_input_h: " << model_input_h << " model_input_w: " << model_input_w);

        // Output bindings (Index 1) [1x5x8400x0]
        detection_attribute_size = engine->getBindingDimensions(1).d[1]; // 5
        num_detections = engine->getBindingDimensions(1).d[2]; // 8400
        num_classes = 1; // Single-class
        LOG("detection_attribute_size: " << detection_attribute_size << " num_detections: " << num_detections << " num_classes: " << num_classes);

        // Allocate buffers for all bindings
        size_t input_size = 3 * model_input_w * model_input_h * sizeof(float);
        LOG("Allocating gpu_buffers[0]: " << input_size << " bytes");
        CUDA_CHECK(cudaMalloc(&gpu_buffers[0], input_size)); // images

        size_t output_size = 5 * 8400 * sizeof(float);
        LOG("Allocating gpu_buffers[1]: " << output_size << " bytes");
        CUDA_CHECK(cudaMalloc(&gpu_buffers[1], output_size)); // outputs

        cpu_output_buffer = new float[detection_attribute_size * num_detections]; // 5 * 8400

        cuda_preprocess_init(MAX_IMAGE_SIZE);
        CUDA_CHECK(cudaStreamCreate(&stream));

        if (warmup) {
            LOG("Starting warmup...");
            Mat dummyFrame(model_input_h, model_input_w, CV_8UC3, Scalar(128, 128, 128));
            for (int i = 0; i < 10; i++) {
                LOG("Warmup iteration " << i + 1);
                preprocess(dummyFrame);
                this->runInference(dummyFrame);
            }
            LOG("model warmup 10 times");
        }
        initialized = true;
    }
 
    DetectionOutput runInference(cv::Mat& frame) override {
        void* buffers[] = { gpu_buffers[0], gpu_buffers[1]};
        context->enqueueV2(buffers, stream, nullptr);
        return DetectionOutput{};
    }
    void releaseOutputs(const DetectionOutput&) override {}
};

Detector* createDetector(const std::string& modelPath, const vector<string>& targetClasses) {
    try {
        return new TensorRTDetector(modelPath, targetClasses);
    } catch (const std::exception& e) {
        std::cerr << "Failed to create detector: " << e.what() << std::endl;
        return nullptr;
    }
}
#include "detector.h"
#include "common.h"
#include "cuda_utils.h"
#include "preprocess.h"
#include <NvOnnxParser.h>
#include "common.h"
#include <fstream>
#include <iostream>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

using namespace nvinfer1;
using namespace std;
using namespace cv;

#define isFP16 true
#define warmup true

// Simple logger implementation for TensorRT
class SimpleLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        
        if (severity <= Severity::kERROR) {
            std::cout << msg << std::endl;
        }
    }
};

class TensorRTDetector : public GenericDetector {
private:
    float* gpu_buffers[2]{};               //!< The vector of device buffers needed for engine execution
    float* cpu_output_buffer{nullptr};     //!< The output buffer for the detection attributes

    cudaStream_t preprocessStream, inferenceStream; //!< CUDA streams for preprocessing and inference
    IRuntime* runtime{nullptr};            //!< The TensorRT runtime used to deserialize the engine
    ICudaEngine* engine{nullptr};          //!< The TensorRT engine used to run the network
    IExecutionContext* context{nullptr};   //!< The context for executing inference using an ICudaEngine

    // Model parameters
    int model_input_w; // Model input width
    int model_input_h; // Model input height
    int num_detections; // Number of detections (e.g. 8400)
    int detection_attribute_size; // Detection attribute size (e.g. 5)
    int num_classes = 1; // Number of classes (e.g. 1, 80 for COCO multi-class)
    const int MAX_IMAGE_SIZE = 1280 * 720; // Maximum image size for preprocessing 
    float conf_threshold = BOX_THRESH; // Confidence threshold
    float nms_threshold = NMS_THRESH; // NMS threshold
    
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
    
    bool buildEngine(const std::string& onnxPath, nvinfer1::ILogger& logger) {
        auto builder = createInferBuilder(logger);
        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
        IBuilderConfig* config = builder->createBuilderConfig();
        if (isFP16) {
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

        if (!engine) {
            std::cout << "Failed to create engine" << std::endl;
            return false;
        } else {
            std::cout << "Successfully created engine" << std::endl;
        }
        return true;
    }

    bool saveEngine(const std::string& onnxpath) {
        std::string engine_path;
        size_t dotIndex = onnxpath.find_last_of(".");
        if (dotIndex != std::string::npos) {
            engine_path = onnxpath.substr(0, dotIndex) + ".engine";
        } else {
            return false;
        }

        if (engine) {
            nvinfer1::IHostMemory* data = engine->serialize();
            std::ofstream file;
            file.open(engine_path, std::ios::binary | std::ios::out);
            if (!file.is_open()) {
                std::cout << "Create engine file " << engine_path << " failed" << std::endl;
                return false;
            }
            file.write((const char*)data->data(), data->size());
            file.close();
            delete data;
        }
        return true;
    }

    void preprocess(cv::Mat& frame) override {
        cuda_preprocess(frame.ptr(), frame.cols, frame.rows, gpu_buffers[0], 
                       model_input_w, model_input_h, preprocessStream);
    }

    void inference(cv::Mat& frame) override {
        CUDA_CHECK(cudaStreamSynchronize(preprocessStream));
        void* buffers[] = {gpu_buffers[0], gpu_buffers[1]};
        context->enqueueV2(buffers, inferenceStream, nullptr);
        CUDA_CHECK(cudaStreamSynchronize(inferenceStream));
    }

    void postprocess(cv::Mat& frame) override {
        detections_.clear(); 
        detections_.reserve(50);

        CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer, gpu_buffers[1], 
                                 detection_attribute_size * num_detections * sizeof(float), 
                                 cudaMemcpyDeviceToHost, inferenceStream));

        const Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);
        
        const float ratio_h = static_cast<float>(model_input_h)/ frame.rows;
        const float ratio_w = static_cast<float>(model_input_w)/ frame.cols; 
        const float scale = std::min(ratio_h, ratio_w);
        const float offset_x = (ratio_h > ratio_w) ? 0.0f : (model_input_w - ratio_h * frame.cols) * 0.5f;
        const float offset_y = (ratio_h > ratio_w) ? (model_input_h - ratio_w * frame.rows) * 0.5f : 0.0f;

        std::vector<Rect> boxes;
        std::vector<float> confidences;
        int class_Id;
        boxes.reserve(50);
        confidences.reserve(50);

        for (int i = 0; i < num_detections; ++i) {
            float confidence = det_output.at<float>(4, i);
            int class_Id = det_output.at<int>(5, i);
            if (confidence > conf_threshold) {
                float cx = det_output.at<float>(0, i);
                float cy = det_output.at<float>(1, i);
                float ow = det_output.at<float>(2, i);
                float oh = det_output.at<float>(3, i);

                Rect box;
                box.x = static_cast<int>((cx - 0.5f * ow - offset_x) / scale);
                box.y = static_cast<int>((cy - 0.5f * oh - offset_y) / scale);
                box.width = static_cast<int>(ow / scale);
                box.height = static_cast<int>(oh / scale);

                boxes.push_back(box);
                confidences.push_back(confidence);
            }
        }

        std::vector<int> nms_result;
        dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_result);

        for (int idx : nms_result) {
            Detection result;
            result.class_id = COCO_LABELS[class_Id];
            result.confidence = confidences[idx];
            result.bounding_box = boxes[idx];
            detections_.push_back(std::move(result));
        }
    }

public:
    TensorRTDetector(const std::string& model_path, const vector<string>& targetClasses)
            : GenericDetector(model_path, targetClasses) {
        static SimpleLogger logger; // Use our custom logger
        if (model_path.find(".onnx") == std::string::npos) {
            initialize(model_path);
        } else {
            buildEngine(model_path, logger);
            saveEngine(model_path);
        }
    }

    ~TensorRTDetector() {
        CUDA_CHECK(cudaStreamSynchronize(preprocessStream));
        CUDA_CHECK(cudaStreamDestroy(preprocessStream));
        CUDA_CHECK(cudaStreamSynchronize(inferenceStream));
        CUDA_CHECK(cudaStreamDestroy(inferenceStream));
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

    void initialize(const string& engineFilePath) override {
        ifstream engineStream(engineFilePath, ios::binary);
        engineStream.seekg(0, ios::end);
        const size_t modelSize = engineStream.tellg();
        engineStream.seekg(0, ios::beg);
        unique_ptr<char[]> engineData(new char[modelSize]);
        engineStream.read(engineData.get(), modelSize);
        engineStream.close();

        static SimpleLogger logger; // Use our custom logger
        runtime = createInferRuntime(logger);
        engine = runtime->deserializeCudaEngine(engineData.get(), modelSize);
        context = engine->createExecutionContext();

        int numBindings = engine->getNbBindings();
        std::cout << "Number of bindings: " << numBindings << std::endl;
        for (int i = 0; i < numBindings; i++) {
            Dims dims = engine->getBindingDimensions(i);
            std::string bindingName = engine->getBindingName(i);
            std::cout << "Binding " << i << ": " << bindingName << " [Dims: " 
                      << dims.d[0] << "x" << dims.d[1] << "x" << dims.d[2] << "x" << dims.d[3] << "]" << std::endl;
        }

        model_input_h = engine->getBindingDimensions(0).d[2];
        model_input_w = engine->getBindingDimensions(0).d[3];
        std::cout << "[INFO] model_input_h: " << model_input_h << " model_input_w: " << model_input_w << std::endl;

        detection_attribute_size = engine->getBindingDimensions(1).d[1];
        num_detections = engine->getBindingDimensions(1).d[2];
        num_classes = 1;
        std::cout << "detection_attribute_size: " << detection_attribute_size 
                  << " num_detections: " << num_detections 
                  << " num_classes: " << num_classes << std::endl;

        size_t input_size = 3 * model_input_w * model_input_h * sizeof(float);
        std::cout << "Allocating gpu_buffers[0]: " << input_size << " bytes" << std::endl;
        CUDA_CHECK(cudaMalloc(&gpu_buffers[0], input_size));

        size_t output_size = 5 * 8400 * sizeof(float);
        std::cout << "Allocating gpu_buffers[1]: " << output_size << " bytes" << std::endl;
        CUDA_CHECK(cudaMalloc(&gpu_buffers[1], output_size));

        cpu_output_buffer = new float[detection_attribute_size * num_detections];

        cuda_preprocess_init(MAX_IMAGE_SIZE);
        CUDA_CHECK(cudaStreamCreate(&preprocessStream));
        CUDA_CHECK(cudaStreamCreate(&inferenceStream));

        if (warmup) {
            std::cout << "Starting warmup..." << std::endl;
            Mat dummyFrame(model_input_h, model_input_w, CV_8UC3, Scalar(128, 128, 128));
            for (int i = 0; i < 10; i++) {
                std::cout << "Warmup iteration " << i + 1 << std::endl;
                preprocess(dummyFrame);
                this->inference(dummyFrame);
            }
            std::cout << "model warmup 10 times" << std::endl;
        }
        initialized_ = true;
    }

    void releaseOutputs() override {
        detections_.clear();    
    }
};

GenericDetector* createDetector(const string& modelPath, const vector<string>& targetClasses) {
    try {
        return new TensorRTDetector(modelPath, targetClasses);
    } catch (const exception& e) {
        std::cerr << "Failed to create detector: " << e.what() << std::endl;
        return nullptr;
    }
}
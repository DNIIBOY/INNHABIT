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
        
    float* gpu_buffers[2]{};               //!< The vector of device buffers needed for engine execution
    float* cpu_output_buffer{nullptr};         //!< The output buffer for the detection attributes

    cudaStream_t preprocessStream, inferenceStream; //!< CUDA streams for preprocessing and inference
    IRuntime* runtime{nullptr};                 //!< The TensorRT runtime used to deserialize the engine
    ICudaEngine* engine{nullptr};               //!< The TensorRT engine used to run the network
    IExecutionContext* context{nullptr};        //!< The context for executing inference using an ICudaEngine

    // Model parameters
    int model_input_w; // Model input width
    int model_input_h; // Model input height
    int num_detections; // Number of detections (e.g. 8400)
    int detection_attribute_size; // Detection attribute size (e.g. 5)
    int num_classes = 1; // Number of classes (e.g. 1, 80 for COCO multi-class)
    const int MAX_IMAGE_SIZE = 1500 * 1500; // Maximum image size for preprocessing 
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

        if (!engine)
        {
            std::cout << "Failed to create engine" << std::endl;
            return false;
        } else {

        }

        std::cout << "Successfully created engine" << std::endl;
        return true;
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
    // call preprocess function in preprocess.cu
    void preprocess(cv::Mat& frame) override {
        // Preprocessing data on gpu
        cuda_preprocess(frame.ptr(), frame.cols, frame.rows, gpu_buffers[0], 
        model_input_w, model_input_h, preprocessStream);
    }
    // call inference function in tensorrt
    void inference(cv::Mat& frame) override {
        CUDA_CHECK(cudaStreamSynchronize(preprocessStream)); // Wait for preprocess
        void* buffers[] = {gpu_buffers[0], gpu_buffers[1]};
        context->enqueueV2(buffers, inferenceStream, nullptr);
    }
    // process outputs from gpu
    void postprocess(cv::Mat& frame) override {
        // Pre-allocate output vector to avoid dynamic resizing
        detections_.clear(); 
        detections_.reserve(50); // Adjust based on expected max detections

        // Synchronize and copy data from GPU to CPU
        CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer, gpu_buffers[1], 
                                detection_attribute_size * num_detections * sizeof(float), 
                                cudaMemcpyDeviceToHost, inferenceStream));
        CUDA_CHECK(cudaStreamSynchronize(inferenceStream));

        // Use a single pass to filter and scale boxes
        const Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);
        
        // Scaling factors and offsets
        const float ratio_h = static_cast<float>(model_input_h)/ frame.rows;
        const float ratio_w = static_cast<float>(model_input_w)/ frame.cols ; 
        const float scale = std::min(ratio_h, ratio_w);
        const float offset_x = (ratio_h > ratio_w) ? 0.0f : (model_input_w - ratio_h * frame.cols) * 0.5f;
        const float offset_y = (ratio_h > ratio_w) ? (model_input_h - ratio_w * frame.rows) * 0.5f : 0.0f;

        // Temporary storage for NMS
        std::vector<Rect> boxes;
        std::vector<float> confidences;
        int class_Id;
        boxes.reserve(50); // Pre-allocate to avoid resizing
        confidences.reserve(50);

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
            result.class_id = COCO_LABELS[class_Id]; // Single-class (use 0 if classId is int)
            result.confidence = confidences[idx];
            result.bounding_box = boxes[idx];
            detections_.push_back(std::move(result)); // Move to avoid copying
        }
    }

public:
    // Constructor with model path and target classes
    TensorRTDetector(const std::string& model_path, const vector<string>& targetClasses)
            : GenericDetector(model_path, targetClasses) {
            if (model_path.find(".onnx") == std::string::npos) {
                initialize(model_path);
            } else {
                buildEngine(model_path, logger);
                saveEngine(model_path);
            }
        }
    // destructor cleanup
    ~TensorRTDetector() {
        // Memory free
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
    // Initialize function to load engine and allocate buffers
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
        CUDA_CHECK(cudaStreamCreate(&preprocessStream));
        CUDA_CHECK(cudaStreamCreate(&inferenceStream));

        if (warmup) {
            LOG("Starting warmup...");
            Mat dummyFrame(model_input_h, model_input_w, CV_8UC3, Scalar(128, 128, 128));
            for (int i = 0; i < 10; i++) {
                LOG("Warmup iteration " << i + 1);
                preprocess(dummyFrame);
                this->inference(dummyFrame);
            }
            LOG("model warmup 10 times");
        }
        preprocess_times_.reserve(100);
        inference_times_.reserve(100);
        postprocess_times_.reserve(100);
        initialized_ = true;
    }
    // dont use just her because using detector.h abstract class i have to change this.
    void releaseOutputs() override {
        // No-op
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
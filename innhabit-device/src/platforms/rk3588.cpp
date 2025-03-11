#include "detector.h"
#include <rknn_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>

using namespace cv;
using namespace std;

class RK3588Detector : public GenericDetector {
private:
    rknn_context ctx;
    rknn_input_output_num io_num;
    vector<rknn_tensor_attr> input_attrs;
    vector<rknn_tensor_attr> output_attrs;
    vector<rknn_output> outputs;
    vector<float> out_scales;
    vector<int32_t> out_zps;

public:
    RK3588Detector(const string& modelPath, const vector<string>& targetClasses_)
        : GenericDetector(modelPath, targetClasses_) {
        initialize(modelPath);
    }

    void detect(Mat& frame) override {
        GenericDetector::detect(frame);  // Call base class detect
        // Buffers are released in releaseOutputs after post_process
    }

protected:
    void initialize(const string& modelPath) override {
        string rknnModel = modelPath + "/yolov7-tiny.rknn";
        int ret = rknn_init(&ctx, (void*)rknnModel.c_str(), 0, 0, NULL);
        if (ret < 0) {
            cerr << "Error: RKNN init failed: " << ret << endl;
            throw runtime_error("RKNN initialization failed");
        }
        ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
        if (ret < 0) {
            rknn_destroy(ctx);
            throw runtime_error("RKNN query failed");
        }
        input_attrs.resize(io_num.n_input);
        for (uint32_t i = 0; i < io_num.n_input; i++) {
            input_attrs[i].index = i;
            ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
            if (ret < 0) {
                rknn_destroy(ctx);
                throw runtime_error("RKNN input query failed");
            }
        }

        output_attrs.resize(io_num.n_output);
        outputs.resize(io_num.n_output);
        out_scales.resize(io_num.n_output);
        out_zps.resize(io_num.n_output);

        for (uint32_t i = 0; i < io_num.n_output; i++) {
            output_attrs[i].index = i;
            ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
            if (ret < 0) {
                rknn_destroy(ctx);
                throw runtime_error("RKNN output query failed");
            }
            outputs[i].index = i;
            outputs[i].want_float = 0;
            out_scales[i] = output_attrs[i].scale;
            out_zps[i] = output_attrs[i].zp;
        }

        if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
            channel = input_attrs[0].dims[1];
            height = input_attrs[0].dims[2];
            width = input_attrs[0].dims[3];
        } else {
            height = input_attrs[0].dims[1];
            width = input_attrs[0].dims[2];
            channel = input_attrs[0].dims[3];
        }
        initialized = true;
    }

    DetectionOutput runInference(const Mat& input) override {
        rknn_input inputs[1];
        memset(inputs, 0, sizeof(inputs));
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].size = width * height * channel;
        inputs[0].fmt = RKNN_TENSOR_NHWC;
        inputs[0].buf = input.data;

        int ret = rknn_inputs_set(ctx, 1, inputs);
        if (ret < 0) {
            cerr << "Error: RKNN inputs set failed: " << ret << endl;
            throw runtime_error("RKNN inputs set failed");
        }

        ret = rknn_run(ctx, NULL);
        if (ret < 0) {
            cerr << "Error: RKNN run failed: " << ret << endl;
            throw runtime_error("RKNN run failed");
        }

        ret = rknn_outputs_get(ctx, io_num.n_output, outputs.data(), NULL);
        if (ret < 0) {
            cerr << "Error: RKNN outputs get failed: " << ret << endl;
            throw runtime_error("RKNN outputs get failed");
        }

        DetectionOutput output;
        output.buffers.resize(io_num.n_output);
        output.scales = out_scales;
        output.zps = out_zps;
        output.num_outputs = io_num.n_output;
        for (uint32_t i = 0; i < io_num.n_output; i++) {
            output.buffers[i] = outputs[i].buf;
        }
        return output;  // Buffers will be released in releaseOutputs
    }

    void releaseOutputs(const DetectionOutput& output) override {
        rknn_outputs_release(ctx, io_num.n_output, outputs.data());
    }

    ~RK3588Detector() override {
        if (initialized) {
            rknn_destroy(ctx);
        }
    }
};

#ifdef USE_RKNN
Detector* createDetector(const string& modelPath, const vector<string>& targetClasses) {
    try {
        return new RK3588Detector(modelPath, targetClasses);
    } catch (const exception& e) {
        cerr << "Error creating RK3588 detector: " << e.what() << endl;
        return nullptr;
    }
}
#else
Detector* createDetector(const string&, const vector<string>&) { return nullptr; }
#endif